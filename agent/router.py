"""
第7课（Part 1）：Agent 路由器与任务规划器

核心接口：
    router = AgentRouter(llm=llm_client, tools=tool_registry)
    
    # 分析用户意图
    intent = router.analyze_intent("对比 A 和 B 产品的差异")
    # 返回: Intent.COMPARISON
    
    # 规划任务
    plan = router.plan(intent, "对比 A 和 B 产品的差异")
    # 返回: ["检索 A 产品参数", "检索 B 产品参数", "对比差异并输出"]
    
    # 执行
    response = router.execute(plan, query="对比 A 和 B 产品的差异")
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from models.document import QueryRequest, ChatResponse
from rag.chains.rag_chain import RAGChain


class Intent(str, Enum):
    """用户意图枚举"""
    FACTUAL_QA = "factual_qa"          # 事实问答（直接 RAG）
    SUMMARIZATION = "summarization"    # 摘要总结
    COMPARISON = "comparison"          # 对比分析
    MULTI_STEP = "multi_step"          # 多步推理
    TOOL_CALL = "tool_call"            # 需要调用外部工具
    CHITCHAT = "chitchat"              # 闲聊（不触发 RAG）


@dataclass
class TaskPlan:
    """任务规划"""
    intent: Intent
    steps: List[str]  # 子任务列表
    requires_tools: List[str]  # 需要调用的工具名
    estimated_complexity: int  # 复杂度 1~5


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str
    input_params: Dict[str, Any]
    output: Any
    success: bool
    error_message: Optional[str] = None


class AgentRouter:
    """
    Agent 路由器

    职责：
    1. 意图识别：判断用户想做什么
    2. 任务规划：复杂任务分解为子步骤
    3. 执行编排：按步骤调用 RAG 或工具，汇总结果
    """

    def __init__(
        self,
        llm: Any,  # LLMClient
        rag_chain: RAGChain,
        tools: Optional[Dict[str, Any]] = None
    ):
        self.llm = llm
        self.rag_chain = rag_chain
        self.tools = tools or {}

    def analyze_intent(self, query: str) -> Intent:
        """
        意图识别

        用 LLM 做零样本分类，比规则匹配更灵活。
        """
        prompt = f"""分析以下用户查询的意图，从以下类别中选择最匹配的一个：

类别：
- factual_qa: 事实性问答（如"什么是 XXX？""怎么操作 YYY？"）
- summarization: 摘要总结（如"总结一下...""这份文档讲了什么"）
- comparison: 对比分析（如"A 和 B 有什么区别""对比两者的优劣"）
- multi_step: 多步推理（如"先...然后...最后..."需要多个步骤才能回答）
- tool_call: 需要外部工具（如"今天天气如何""查一下数据库"）
- chitchat: 闲聊问候（如"你好""谢谢"）

用户查询：{query}

只回答意图类别名称，不要解释："""

        response = self.llm.generate(
            system_prompt="你是一个意图分类助手。只输出类别名称。",
            user_prompt=prompt,
            temperature=0.1
        ).strip().lower()

        # 映射到枚举
        intent_map = {
            "factual_qa": Intent.FACTUAL_QA,
            "summarization": Intent.SUMMARIZATION,
            "comparison": Intent.COMPARISON,
            "multi_step": Intent.MULTI_STEP,
            "tool_call": Intent.TOOL_CALL,
            "chitchat": Intent.CHITCHAT,
        }

        return intent_map.get(response, Intent.FACTUAL_QA)

    def plan(self, intent: Intent, query: str) -> TaskPlan:
        """
        根据意图生成执行计划
        """
        if intent == Intent.FACTUAL_QA:
            return TaskPlan(
                intent=intent,
                steps=["直接检索相关知识并回答"],
                requires_tools=["rag_retrieval"],
                estimated_complexity=1
            )

        elif intent == Intent.SUMMARIZATION:
            return TaskPlan(
                intent=intent,
                steps=[
                    "检索相关文档的全部内容",
                    "提取关键要点",
                    "组织成连贯的摘要"
                ],
                requires_tools=["rag_retrieval", "context_compressor"],
                estimated_complexity=3
            )

        elif intent == Intent.COMPARISON:
            return TaskPlan(
                intent=intent,
                steps=[
                    "识别比较对象（如 A 和 B）",
                    "分别检索每个对象的详细信息",
                    "提取可对比的维度",
                    "生成对比表格或分点总结"
                ],
                requires_tools=["rag_retrieval"],
                estimated_complexity=4
            )

        elif intent == Intent.MULTI_STEP:
            # 用 LLM 动态分解任务
            return self._dynamic_plan(query)

        elif intent == Intent.TOOL_CALL:
            return TaskPlan(
                intent=intent,
                steps=["识别需要调用的工具", "执行工具调用", "整合结果回答"],
                requires_tools=["dynamic_tool"],
                estimated_complexity=3
            )

        elif intent == Intent.CHITCHAT:
            return TaskPlan(
                intent=intent,
                steps=["直接友好回复"],
                requires_tools=[],
                estimated_complexity=1
            )

        return TaskPlan(
            intent=Intent.FACTUAL_QA,
            steps=["直接检索相关知识并回答"],
            requires_tools=["rag_retrieval"],
            estimated_complexity=1
        )

    def _dynamic_plan(self, query: str) -> TaskPlan:
        """
        动态任务分解（针对复杂多步问题）

        让 LLM 自己把问题拆成可执行的子步骤。
        """
        prompt = f"""将以下复杂问题分解为最多 5 个可执行的子步骤。
每个步骤应该清晰明确，可以独立执行。

问题：{query}

请按顺序列出步骤（每行一个，不要编号）："""

        response = self.llm.generate(
            system_prompt="你是一个任务规划助手。",
            user_prompt=prompt,
            temperature=0.3
        )

        steps = [s.strip() for s in response.strip().split("\n") if s.strip()]
        steps = steps[:5]  # 最多 5 步

        return TaskPlan(
            intent=Intent.MULTI_STEP,
            steps=steps,
            requires_tools=["rag_retrieval"],
            estimated_complexity=len(steps)
        )

    def execute(self, plan: TaskPlan, query: str) -> ChatResponse:
        """
        执行规划好的任务

        根据意图类型选择不同的执行策略。
        """
        if plan.intent == Intent.FACTUAL_QA:
            # 直接走 RAG 链
            return self.rag_chain.invoke(QueryRequest(query=query))

        elif plan.intent == Intent.COMPARISON:
            return self._execute_comparison(query)

        elif plan.intent == Intent.SUMMARIZATION:
            return self._execute_summarization(query)

        elif plan.intent == Intent.MULTI_STEP:
            return self._execute_multi_step(plan, query)

        elif plan.intent == Intent.CHITCHAT:
            return self._execute_chitchat(query)

        else:
            # 默认走 RAG
            return self.rag_chain.invoke(QueryRequest(query=query))

    def _execute_comparison(self, query: str) -> ChatResponse:
        """
        执行对比任务

        1. 用 LLM 提取比较对象
        2. 分别检索每个对象
        3. 用 LLM 生成对比结果
        """
        # 提取比较对象
        extract_prompt = f"""从以下查询中提取需要对比的对象名称（最多2个）：

查询：{query}

只输出对象名称，每行一个："""

        response = self.llm.generate(
            system_prompt="提取对比对象。",
            user_prompt=extract_prompt,
            temperature=0.1
        )
        objects = [s.strip() for s in response.strip().split("\n") if s.strip()][:2]

        if len(objects) < 2:
            # 回退到普通 RAG
            return self.rag_chain.invoke(QueryRequest(query=query))

        # 分别检索每个对象
        all_sources = []
        object_contexts = []

        for obj in objects:
            obj_query = f"{obj} 的参数 规格 功能"
            result = self.rag_chain.invoke(QueryRequest(query=obj_query, top_k=3))
            context = f"\n=== {obj} 的相关信息 ===\n{result.answer}\n"
            object_contexts.append(context)
            all_sources.extend(result.sources)

        # 生成对比
        combined_context = "\n".join(object_contexts)
        compare_prompt = f"""基于以下信息，对比 {objects[0]} 和 {objects[1]} 的差异：

{combined_context}

请从多个维度（如功能、性能、价格、适用场景等）进行对比，用表格或分点形式输出。"""

        answer = self.llm.generate(
            system_prompt="你是一个专业的产品对比分析师。",
            user_prompt=compare_prompt,
            temperature=0.3
        )

        return ChatResponse(
            answer=answer,
            sources=all_sources,
            query_time_ms=None
        )

    def _execute_summarization(self, query: str) -> ChatResponse:
        """执行摘要任务"""
        # 先广泛检索
        result = self.rag_chain.invoke(QueryRequest(query=query, top_k=10))

        if not result.sources:
            return result

        # 使用上下文压缩器做 Map-Reduce 摘要
        from rag.chains.rag_chain import ContextCompressor
        compressor = ContextCompressor(self.llm)
        summary = compressor.compress(query, result.sources)

        return ChatResponse(
            answer=summary,
            sources=result.sources,
            query_time_ms=result.query_time_ms
        )

    def _execute_multi_step(self, plan: TaskPlan, query: str) -> ChatResponse:
        """
        执行多步推理任务

        按步骤逐个执行，前一步的结果作为后一步的上下文。
        """
        step_results = []
        all_sources = []

        for i, step in enumerate(plan.steps):
            # 构建当前步骤的查询（结合历史结果）
            context = "\n".join(step_results) if step_results else "无"
            step_prompt = f"""执行以下步骤，基于之前的结果：

原始问题：{query}
已完成步骤结果：
{context}

当前步骤：{step}

请执行此步骤并输出结果："""

            # 使用 RAG 获取当前步骤的相关知识
            rag_result = self.rag_chain.invoke(QueryRequest(query=step))
            all_sources.extend(rag_result.sources)

            # 让 LLM 整合 RAG 结果和步骤指令
            final_prompt = f"""步骤：{step}

参考资料：
{rag_result.answer}

请完成此步骤："""

            step_answer = self.llm.generate(
                system_prompt="你是一个执行助手，按步骤完成任务。",
                user_prompt=final_prompt,
                temperature=0.3
            )

            step_results.append(f"步骤 {i+1} ({step}): {step_answer}")

        # 汇总所有步骤结果，生成最终答案
        step_results_text = "\n".join(step_results)
        final_summary_prompt = f"""基于以下各步骤的执行结果，回答用户的原始问题：

原始问题：{query}

各步骤结果：
{step_results_text}

请给出最终答案："""

        final_answer = self.llm.generate(
            system_prompt="你是一个综合分析师，汇总多步推理的结果。",
            user_prompt=final_summary_prompt,
            temperature=0.3
        )

        return ChatResponse(
            answer=final_answer,
            sources=all_sources
        )

    def _execute_chitchat(self, query: str) -> ChatResponse:
        """执行闲聊回复"""
        answer = self.llm.generate(
            system_prompt="你是一个友好的 AI 助手。",
            user_prompt=query,
            temperature=0.7
        )
        return ChatResponse(answer=answer, sources=[])
