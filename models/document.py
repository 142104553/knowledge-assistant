"""
第1课：数据模型定义

所有模块共享的数据结构，使用 Pydantic 进行类型校验。
Pydantic 的优势：自动类型转换、数据验证、JSON 序列化。
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """支持的文档类型"""
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    MD = "markdown"
    HTML = "html"
    TXT = "txt"
    CSV = "csv"


class DocumentChunk(BaseModel):
    """
    文本分块后的单元
    
    每个 chunk 包含：
    - content: 实际文本内容
    - metadata: 来源信息（文件名、页码、章节等）
    - embedding: 向量化后的结果（可选，离线阶段生成）
    """
    content: str = Field(description="文本块内容")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据：文件名、页码、章节、 chunk 索引等"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="向量表示（由 Embedding 模型生成）"
    )

    class Config:
        # 允许从 ORM 对象等属性创建
        from_attributes = True


class SourceDocument(BaseModel):
    """
    原始文档的元信息
    
    记录文档的基本信息，用于溯源和展示。
    """
    doc_id: str = Field(description="文档唯一标识（通常用 UUID 或文件哈希）")
    filename: str = Field(description="原始文件名")
    doc_type: DocumentType = Field(description="文档类型")
    file_path: str = Field(description="文件存储路径")
    total_pages: Optional[int] = Field(default=None, description="总页数（如适用）")
    created_at: datetime = Field(default_factory=datetime.now, description="入库时间")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="额外元数据：作者、创建日期、标签等"
    )


class RetrievedChunk(DocumentChunk):
    """
    检索返回的文本块（继承自 DocumentChunk）
    
    相比普通 chunk，多了相似度分数，用于排序和阈值过滤。
    """
    score: float = Field(description="相似度分数（由向量检索或重排序模型给出）")
    rank: Optional[int] = Field(default=None, description="重排序后的位次")


class QueryRequest(BaseModel):
    """
    用户查询请求
    
    这是进入系统的入口数据结构。
    """
    query: str = Field(description="用户原始问题")
    session_id: Optional[str] = Field(default=None, description="对话 session ID，用于多轮对话")
    top_k: int = Field(default=5, description="召回文档数量", ge=1, le=20)
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="元数据过滤条件，如 {'doc_type': 'pdf', 'author': '张三'}"
    )
    # 第7课会用到：是否启用 Agent 进行任务分解
    enable_agent: bool = Field(default=False, description="是否启用 Agent 模式")


class ChatResponse(BaseModel):
    """
    系统回答
    
    包含：生成的答案、引用的来源、处理耗时等。
    """
    answer: str = Field(description="LLM 生成的回答")
    sources: List[RetrievedChunk] = Field(
        default_factory=list,
        description="回答所引用的文档片段（用于溯源）"
    )
    query_time_ms: Optional[int] = Field(default=None, description="查询耗时（毫秒）")
    session_id: Optional[str] = Field(default=None, description="对话 session ID")


# ── 第1课小结 ──
# 
# 你学到了：
# 1. RAG 的核心是 "检索 + 生成"，解决 LLM 知识幻觉问题
# 2. Agent 让系统从"问答"升级为"任务执行"
# 3. 系统分为离线（ingestion）和在线（query）两条数据流
# 4. Pydantic BaseModel 是 Python 项目中定义数据契约的最佳实践
#
# 下节课：我们将进入 ingestion 管道，学习如何解析 PDF、Word 等多格式文档。
