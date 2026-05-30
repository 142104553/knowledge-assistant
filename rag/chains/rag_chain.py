"""
第6课（Part 3）：RAG 主链组装

把前面所有组件串联成完整的问答流程：
    用户问题 → 查询分析 → 混合检索 → 重排序 → 上下文组装 → LLM 生成 → 带溯源的回答

核心接口：
    chain = RAGChain(
        embedder=embedding_client,
        retriever=hybrid_retriever,
        reranker=cross_encoder_reranker,
        llm=llm_client
    )
    response = chain.invoke("如何申请退款？")
"""

from typing import List, Optional
from datetime import datetime

from models.document import QueryRequest, ChatResponse, RetrievedChunk
from embeddings.factory import BaseEmbeddingClient
from rag.retrievers.hybrid import HybridRetriever
from rag.post_processors.reranker import BaseReranker


class LLMClient:
    """
    LLM 客户端（简化版，实际项目中可扩展为工厂模式）

    封装 OpenAI / Azure / 本地模型等不同的 LLM 调用方式。
    这里以 OpenAI 为例。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model

        # LLM 客户端初始化完成

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = openai.OpenAI(**client_kwargs)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3
    ) -> str:
        """调用 LLM 生成文本"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content


class RAGChain:
    """
    RAG 问答链

    这是整个 RAG 层的核心编排类，协调 Embedding、检索、重排、生成四个阶段。
    """

    # 系统提示词：约束 LLM 的行为
    SYSTEM_PROMPT = """你是一个基于知识库的问答助手，同时具备电力系统分析专家的推理能力。

回答规则：
1. 【忠实检索】你只能根据提供的「参考资料」回答问题，禁止编造知识库中没有的信息。
2. 【跨文档分析】当参考资料来自多个不同文件时，你必须：
   - 对比不同文件中的关键参数（数值、阈值、标准）
   - 找出它们之间的因果关系、约束关系或冲突点
   - 不要简单罗列每个文件的内容，而要回答"这些因素如何相互作用"
3. 【数值敏感】遇到电流、电压、时间、百分比、容量等数值时，必须做大小比较和裕度分析，指出"是否满足要求"、"有无安全余量"。
4. 【核心洞察】在回答末尾，用一句话提炼本次分析的核心教训或工程启示（如果适用）。
5. 【引用规范】引用来源时注明文件名和页码，格式如「根据 保护误动事故案例分析 第1页」。
6. 【边界判定】只有当参考资料完全与问题无关时，才回答："根据现有知识库，无法找到相关信息。"
"""

    def __init__(
        self,
        embedder: BaseEmbeddingClient,
        retriever: HybridRetriever,
        reranker: BaseReranker,
        llm: LLMClient,
        max_context_tokens: int = 4000
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.max_context_tokens = max_context_tokens

    def invoke(self, query_request: QueryRequest) -> ChatResponse:
        """
        执行完整的 RAG 问答流程

        Args:
            query_request: 包含用户查询、top_k、过滤条件等

        Returns:
            ChatResponse: 包含回答、引用来源、耗时等
        """
        start_time = datetime.now()
        stage_times = {}

        # === 阶段 1：查询向量化 ===
        t0 = datetime.now()
        query_embedding = self.embedder.embed([query_request.query])[0]
        stage_times['embed'] = int((datetime.now() - t0).total_seconds() * 1000)

        # === 阶段 2：混合检索（召回） ===
        t0 = datetime.now()
        candidates = self.retriever.retrieve(
            query=query_request.query,
            query_embedding=query_embedding,
            top_k=query_request.top_k * 3,  # 多召回一些给重排
            filter_dict=query_request.filters
        )
        stage_times['retrieve'] = int((datetime.now() - t0).total_seconds() * 1000)

        # === 阶段 3：重排序（精排） ===
        t0 = datetime.now()
        ranked = self.reranker.rerank(
            query=query_request.query,
            candidates=candidates,
            top_n=query_request.top_k
        )
        stage_times['rerank'] = int((datetime.now() - t0).total_seconds() * 1000)

        # === 阶段 4：上下文压缩与组装 ===
        t0 = datetime.now()
        context = self._build_context(ranked)
        stage_times['context'] = int((datetime.now() - t0).total_seconds() * 1000)

        # === 阶段 5：LLM 生成 ===
        t0 = datetime.now()
        if not ranked:
            # 空结果触发拒答
            answer = "根据现有知识库，无法找到与您的提问相关的信息。"
        else:
            user_prompt = self._build_prompt(query_request.query, context, ranked)
            answer = self.llm.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt
            )
        stage_times['llm'] = int((datetime.now() - t0).total_seconds() * 1000)

        elapsed = int((datetime.now() - start_time).total_seconds() * 1000)
        print(f"[RAG耗时] 总:{elapsed}ms | embed:{stage_times['embed']}ms retrieve:{stage_times['retrieve']}ms rerank:{stage_times['rerank']}ms context:{stage_times['context']}ms llm:{stage_times['llm']}ms | chunks:{len(ranked)} | query_len:{len(query_request.query)} | context_len:{len(context)}")

        return ChatResponse(
            answer=answer,
            sources=ranked,
            query_time_ms=elapsed,
            session_id=query_request.session_id
        )

    def _build_context(self, chunks: List[RetrievedChunk]) -> tuple[str, set[str]]:
        """
        将检索到的 chunk 组装成上下文字符串

        策略：
        1. 按文件来源分组，确保每个文件至少有一个 chunk 进入上下文
        2. 全局按分数排序，确保截断时保留最高分的 chunk
        3. 累计长度接近上限时截断

        Returns:
            (context_str, files_in_context): 组装后的上下文字符串，以及实际进入上下文的文件集合
        """
        if not chunks:
            return "", set()

        chars_limit = int((self.max_context_tokens - 400) / 1.5)  # 预留 system_prompt + query 的 token 预算

        # 按文件分组，每个文件取最高分的 chunk 优先
        from collections import defaultdict
        file_groups = defaultdict(list)
        for chunk in chunks:
            filename = chunk.metadata.get('source_file', '未知')
            file_groups[filename].append(chunk)

        # 每个文件先取 Top-1（确保多文件覆盖）
        selected = []
        for filename, group in file_groups.items():
            group.sort(key=lambda x: x.score, reverse=True)
            selected.append(group[0])  # 每个文件至少一个

        # 补充剩余的高分 chunk（去重）
        seen = {id(c) for c in selected}
        for chunk in chunks:
            if id(chunk) not in seen:
                selected.append(chunk)
                seen.add(id(chunk))

        # 全局按分数排序，确保截断时保留高分 chunk
        selected.sort(key=lambda x: x.score, reverse=True)

        # 组装上下文
        context_parts = []
        current_length = 0
        files_in_context = set()

        for chunk in selected:
            filename = chunk.metadata.get('source_file', '未知文件')
            page_raw = chunk.metadata.get('page_number', chunk.metadata.get('page_index', 'N/A'))
            # 处理 0-based page_index
            if isinstance(page_raw, int) and 'page_index' in chunk.metadata and 'page_number' not in chunk.metadata:
                page = page_raw + 1
            else:
                page = page_raw
            source_label = f"【{filename} 第{page}页】"

            part = f"\n--- {source_label} ---\n{chunk.content}\n"
            part_length = len(part)

            if current_length + part_length > chars_limit and context_parts:
                break

            context_parts.append(part)
            current_length += part_length
            files_in_context.add(filename)

        return "".join(context_parts), files_in_context

    def _build_prompt(
        self,
        query: str,
        context: str,
        chunks: List[RetrievedChunk],
        files_in_context: set[str] = None
    ) -> str:
        """
        构建发送给 LLM 的用户提示词

        引用格式要求：直接引用「文件名 第X页」，不要只用数字编号。
        """
        # 检测是否多文件来源（基于实际进入 context 的文件，而非全部 ranked chunks）
        unique_files = files_in_context if files_in_context is not None else set()

        analysis_instruction = ""
        if len(unique_files) >= 2:
            analysis_instruction = """
【分析要求】以上资料来自多个文件。回答时请注意：
- 如果不同文件给出了相互关联的参数，请做对比分析（哪个是约束条件、哪个是实际值、是否满足要求）
- 如果问题是"为什么..."，请找出跨文档的因果链，而不仅是分别陈述两个事实
- 如果涉及工程判断，请指出"如果只看A会得出什么结论，但B告诉我们另一个事实"
"""

        prompt = f"""用户问题：{query}

=== 参考资料 ===
{context}
{analysis_instruction}

请根据以上参考资料回答用户问题。引用时请直接注明来源，如「根据 保护误动事故案例分析 第1页...」。如果资料中没有相关信息，请明确说明无法找到答案。
"""
        return prompt


class ContextCompressor:
    """
    上下文压缩器（进阶功能）

    当检索结果总长度远超 LLM 上下文窗口时，使用 Map-Reduce 策略：
    1. Map：让每个 chunk 独立生成一个"要点摘要"
    2. Reduce：把所有要点汇总，作为最终上下文

    适用场景：用户问题需要浏览大量文档（如"总结这份报告的所有风险点"）
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def compress(self, query: str, chunks: List[RetrievedChunk]) -> str:
        """
        Map-Reduce 压缩
        """
        # Map 阶段：每个 chunk 生成一个相关要点
        map_prompt = """给定用户问题和一段参考资料，提取与问题相关的关键要点。
如果参考资料与问题无关，回答"无关"。

用户问题：{query}

参考资料：
{content}

相关要点（最多3条）："""

        all_points = []
        for chunk in chunks:
            prompt = map_prompt.format(query=query, content=chunk.content)
            response = self.llm.generate(
                system_prompt="你是一个信息提取助手。",
                user_prompt=prompt,
                temperature=0.1
            )
            if "无关" not in response:
                all_points.append(response.strip())

        # Reduce 阶段：汇总要点
        if not all_points:
            return "无相关资料"

        combined = "\n".join([f"- {p}" for p in all_points])
        reduce_prompt = f"""将以下要点整理成连贯的上下文摘要：

{combined}

整理后的摘要："""

        summary = self.llm.generate(
            system_prompt="你是一个文本摘要助手。",
            user_prompt=reduce_prompt,
            temperature=0.2
        )

        return summary


# ── 第6课小结 ──
#
# 你学到了：
# 1. 混合检索：Dense（向量）+ Sparse（BM25）互补，覆盖语义和关键词匹配
# 2. RRF 融合：用排名位置而非绝对分数来合并两种检索结果
# 3. Cross-Encoder 重排序：精度更高，但只对 Top-K 候选做
# 4. 上下文压缩：分数过滤 + 长度截断 + Map-Reduce 摘要
# 5. 系统提示词工程：明确约束 LLM"不编造"，实现拒答机制
#
# 思考题：
# - 为什么 Cross-Encoder 比 Bi-Encoder（向量检索）精度更高？
#   （答案：Bi-Encoder 分别编码 query 和 doc，只能计算向量相似度；
#   Cross-Encoder 把 query 和 doc 一起输入模型，模型可以看到两者的词级别交互，
#   所以能判断"苹果的 CEO 是谁"和"苹果公司简介"的相关性更高。）
#
# 下节课：Agent 架构 —— 当用户的问题不是简单问答，而是需要多步推理和工具调用时怎么办？
