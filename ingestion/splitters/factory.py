"""
第3课：文本分块策略

核心接口：
    splitter = get_splitter(strategy="recursive", chunk_size=512, overlap=50)
    chunks = splitter.split(document_chunks)

所有分块器接收 List[DocumentChunk]，返回 List[DocumentChunk]（更细的粒度）。
"""

import re
from abc import ABC, abstractmethod
from typing import List

from models.document import DocumentChunk


class BaseSplitter(ABC):
    """
    分块器抽象基类
    
    子类需要实现 split 方法，将输入的 DocumentChunk 切成更小的块。
    注意：输入可能是整页内容（来自 loader），也可能是已经很大的文本块。
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: 目标块大小（按字符数计，中文字符≈token，可近似）
            chunk_overlap: 相邻块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        pass

    def _merge_small_chunks(
        self,
        chunks: List[DocumentChunk],
        min_size: int = 50
    ) -> List[DocumentChunk]:
        """
        合并过小的碎片（比如单个标点、页眉残留）
        
        策略：如果当前块小于 min_size，尝试与下一个块合并。
        """
        if not chunks:
            return []

        merged = []
        buffer = chunks[0]

        for chunk in chunks[1:]:
            if len(buffer.content) < min_size:
                # 合并到 buffer，元数据保留 buffer 的（简化处理）
                buffer = DocumentChunk(
                    content=buffer.content + "\n" + chunk.content,
                    metadata=buffer.metadata,
                )
            else:
                merged.append(buffer)
                buffer = chunk

        merged.append(buffer)
        return merged


class RecursiveCharacterSplitter(BaseSplitter):
    """
    递归字符分块器（推荐作为默认策略）
    
    原理：按分隔符的优先级逐级尝试切分
    优先级（中文适配版）：
        1. 段落分隔符（\n\n）
        2. 换行符（\n）
        3. 中文句号（。）
        4. 英文句号 + 空格（. ）
        5. 中文逗号（，）
        6. 空格（ ）
        7. 字符级（兜底）
    
    为什么叫"递归"？
    如果一级分隔符切出来的块仍然大于 chunk_size，就用二级分隔符继续切，
    直到每个块都小于 chunk_size。
    """

    # 分隔符优先级（从高到低）
    DEFAULT_SEPARATORS = [
        "\n\n",     # 段落
        "\n",       # 换行
        "。",       # 中文句号
        "\. ",      # 英文句号+空格（正则需要转义）
        "，",       # 中文逗号
        " ",        # 空格
        "",         # 字符级（空字符串表示逐字符）
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        result = []
        for chunk in chunks:
            split_chunks = self._split_text(chunk.content, chunk.metadata)
            result.extend(split_chunks)
        return self._merge_small_chunks(result)

    def _split_text(
        self,
        text: str,
        metadata: dict,
        separator_index: int = 0
    ) -> List[DocumentChunk]:
        """
        递归切分文本
        
        这是整个分块策略的核心算法。
        """
        # 终止条件：如果文本已经够短，直接返回
        if len(text) <= self.chunk_size:
            return [DocumentChunk(content=text, metadata=metadata)]

        # 如果所有分隔符都试过了，强行按字符切
        if separator_index >= len(self.separators):
            return self._hard_split(text, metadata)

        separator = self.separators[separator_index]

        if separator == "":
            # 字符级切分：严格按 chunk_size，保留 overlap
            return self._hard_split(text, metadata)

        # 用正则分割，保留分隔符（把分隔符附到前一段）
        parts = re.split(f"({re.escape(separator)})", text)
        # re.split 会返回分隔符作为独立元素，需要合并：["段落1", "\n\n", "段落2"]
        merged_parts = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i + 1] == separator:
                merged_parts.append(parts[i] + parts[i + 1])
                i += 2
            else:
                merged_parts.append(parts[i])
                i += 1

        # 贪心组装：尽量把连续的 part 合并到接近 chunk_size
        current_chunk = ""
        result = []

        for part in merged_parts:
            if not part:
                continue

            if len(current_chunk) + len(part) <= self.chunk_size:
                current_chunk += part
            else:
                if current_chunk:
                    result.append(DocumentChunk(
                        content=current_chunk.strip(),
                        metadata=metadata
                    ))
                # 处理单个 part 就超过 chunk_size 的情况（需要更深递归）
                if len(part) > self.chunk_size:
                    # 用下一级分隔符继续切这个 part
                    deeper = self._split_text(
                        part, metadata, separator_index + 1
                    )
                    # 把 deeper 的结果放入 result，但要处理 overlap
                    result.extend(deeper)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            result.append(DocumentChunk(
                content=current_chunk.strip(),
                metadata=metadata
            ))

        # 添加 overlap：每个 chunk（除了第一个）前面追加前一个 chunk 的尾部
        if self.chunk_overlap > 0 and len(result) > 1:
            result = self._add_overlap(result)

        return result

    def _hard_split(self, text: str, metadata: dict) -> List[DocumentChunk]:
        """强制按字符切分，严格遵循 chunk_size 和 overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(DocumentChunk(
                content=chunk_text,
                metadata=metadata
            ))
            # 下一步的起始位置要考虑 overlap
            start = end - self.chunk_overlap
            if start <= 0:  # 防止死循环
                start = end
        return chunks

    def _add_overlap(
        self,
        chunks: List[DocumentChunk]
    ) -> List[DocumentChunk]:
        """
        在相邻 chunk 之间添加重叠区域
        
        策略：第 N 个 chunk 的前面加上第 N-1 个 chunk 的最后 overlap 个字符
        """
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].content
            overlap_text = prev_text[-self.chunk_overlap:] if len(prev_text) > self.chunk_overlap else prev_text
            new_content = overlap_text + chunks[i].content
            result.append(DocumentChunk(
                content=new_content,
                metadata=chunks[i].metadata
            ))
        return result


class MarkdownHeaderSplitter(BaseSplitter):
    """
    Markdown 标题感知分块器
    
    原理：以 Markdown 标题（# ## ###）作为分块边界，
    每个 chunk 包含一个标题及其下属内容。
    
    优势：保留了文档的层级结构，检索时"第四章-第二节"这样的标题
    本身就是强语义信号。
    """

    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        # Markdown 分块通常不需要 overlap，因为标题边界已经很清晰
        super().__init__(chunk_size, chunk_overlap)

    def split(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        result = []
        for chunk in chunks:
            # 只对 Markdown 文件使用此策略
            if chunk.metadata.get("doc_type") != "markdown":
                # 回退到递归分块
                fallback = RecursiveCharacterSplitter(
                    self.chunk_size, self.chunk_overlap
                )
                result.extend(fallback.split([chunk]))
                continue

            split_chunks = self._split_by_headers(
                chunk.content, chunk.metadata
            )
            result.extend(split_chunks)
        return result

    def _split_by_headers(
        self,
        text: str,
        metadata: dict
    ) -> List[DocumentChunk]:
        """按 Markdown 标题切分"""
        matches = list(self.HEADER_PATTERN.finditer(text))
        
        if not matches:
            # 没有标题，回退到递归分块
            fallback = RecursiveCharacterSplitter(
                self.chunk_size, self.chunk_overlap
            )
            return fallback.split([DocumentChunk(content=text, metadata=metadata)])

        result = []
        for i, match in enumerate(matches):
            header_level = len(match.group(1))  # # 的数量
            header_text = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_text = text[start:end].strip()
            
            # 如果 section 太长，用递归分块二次切分
            if len(section_text) > self.chunk_size:
                fallback = RecursiveCharacterSplitter(
                    self.chunk_size, self.chunk_overlap
                )
                sub_chunks = fallback.split([
                    DocumentChunk(content=section_text, metadata=metadata)
                ])
                # 给第一个 sub_chunk 加上标题信息
                if sub_chunks:
                    sub_chunks[0].metadata["header"] = header_text
                    sub_chunks[0].metadata["header_level"] = header_level
                result.extend(sub_chunks)
            else:
                new_metadata = {
                    **metadata,
                    "header": header_text,
                    "header_level": header_level,
                }
                result.append(DocumentChunk(
                    content=section_text,
                    metadata=new_metadata
                ))

        return result


class SemanticSplitter(BaseSplitter):
    """
    语义分块器（高级策略，了解即可）
    
    原理：计算相邻句子的 Embedding 相似度，
    当相似度低于阈值时，认为此处发生了"话题切换"，在此切分。
    
    优势：切分点真正在语义边界上
    劣势：每个 chunk 都需要调用 Embedding API，成本高、速度慢
    
    实际使用建议：
    - 对质量要求极高的场景（如法律合同分析）
    - 可以先用 RecursiveSplitter 粗分，再用 SemanticSplitter 微调边界
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.85
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        # 第4课会实现 EmbeddingClient，这里先预留接口
        self._embedding_client = None

    def split(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        # 语义分块需要 Embedding 服务，在 factory 中注入
        raise NotImplementedError(
            "SemanticSplitter 需要先配置 EmbeddingClient（第4课内容）"
        )


# ═══════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════

def get_splitter(
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> BaseSplitter:
    """
    获取分块器实例
    
    Args:
        strategy: "recursive" | "markdown" | "semantic"
        chunk_size: 块大小（字符数）
        chunk_overlap: 重叠字符数
    """
    if strategy == "recursive":
        return RecursiveCharacterSplitter(chunk_size, chunk_overlap)
    elif strategy == "markdown":
        return MarkdownHeaderSplitter(chunk_size, chunk_overlap)
    elif strategy == "semantic":
        return SemanticSplitter(chunk_size, chunk_overlap)
    else:
        raise ValueError(f"未知的分块策略: {strategy}")


# ── 第3课小结 ──
#
# 你学到了：
# 1. 分块的必要性：解决上下文窗口限制和检索精度问题
# 2. 递归字符分块：按段落→句子→单词的优先级逐级切分
# 3. Markdown 标题分块：利用文档结构保留层级语义
# 4. 语义分块：用 Embedding 相似度找话题切换点（成本高但精准）
#
# 思考题：
# - chunk_size 越大，召回率越高还是越低？（答案：chunk_size 越大，每个 chunk 包含的信息越多，
#   但向量表示越模糊，精准匹配能力下降。需要实验调优。）
#
# 下节课：Embedding —— 如何把"文本块"变成"数学向量"？
