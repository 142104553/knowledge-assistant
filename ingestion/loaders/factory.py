"""
第2课：统一文档加载器工厂

核心思想：无论用户上传 PDF、Word 还是 PPT，调用方只关心一个接口：
    chunks = load_document(file_path)

返回的永远是 List[DocumentChunk]，上层模块无需关心原始格式。
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

# 第4课会详细讲，这里先了解：Document 是 LangChain 的标准文档单元
from langchain.schema import Document as LangChainDocument

# models/document.py 中定义的 Pydantic 模型
from models.document import DocumentChunk, DocumentType, SourceDocument


class BaseDocumentLoader(ABC):
    """
    所有文档加载器的抽象基类。
    
    如果你以后要支持新的格式（比如 EPUB），只需要：
    1. 继承 BaseDocumentLoader
    2. 实现 load() 方法
    3. 在 DocumentLoaderFactory 中注册
    """

    @abstractmethod
    def load(self, file_path: str) -> List[DocumentChunk]:
        """
        加载文档并返回标准化的文本块列表。
        
        Args:
            file_path: 文件的绝对路径或相对路径
            
        Returns:
            List[DocumentChunk]: 每个 chunk 包含内容和元数据
        """
        pass

    def _make_chunks_from_pages(
        self,
        pages: List[LangChainDocument],
        filename: str,
        doc_type: DocumentType
    ) -> List[DocumentChunk]:
        """
        通用辅助方法：将 LangChain Document 列表转换为我们定义的 DocumentChunk。
        
        LangChain 的 Document 结构：
        - page_content: str  (文本内容)
        - metadata: dict    (页码、来源等)
        """
        chunks = []
        for i, page in enumerate(pages):
            # 构建元数据：保留原始信息，同时统一添加我们的字段
            metadata = {
                **page.metadata,           # 原始元数据（如页码）
                "source_file": filename,   # 来源文件名
                "doc_type": doc_type.value,# 文档类型
                "page_index": i,           # 在文档中的序号
            }
            chunks.append(DocumentChunk(
                content=page.page_content,
                metadata=metadata
            ))
        return chunks


class PDFLoader(BaseDocumentLoader):
    """
    PDF 文档加载器
    
    技术选型：PyMuPDF (fitz)
    优势：速度快，能提取页面坐标信息（用于后续判断是否为双栏布局）
    劣势：对扫描版 PDF 需要 OCR（可后续接入 pytesseract）
    """

    def load(self, file_path: str) -> List[DocumentChunk]:
        # 延迟导入：只在真正加载 PDF 时才引入 fitz，减少启动耗时
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF 未安装。请运行：pip install pymupdf"
            )

        doc = fitz.open(file_path)
        filename = Path(file_path).name
        pages = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # extract_text() 会自动处理多栏布局，按阅读顺序输出
            text = page.get_text("text")
            
            # 清理常见噪音（第2课暂不深入，后续可在 cleaners/ 中扩展）
            text = self._clean_text(text)
            
            pages.append(LangChainDocument(
                page_content=text,
                metadata={
                    "page_number": page_num + 1,
                    "total_pages": len(doc),
                }
            ))

        doc.close()
        return self._make_chunks_from_pages(
            pages, filename, DocumentType.PDF
        )

    def _clean_text(self, text: str) -> str:
        """基础文本清洗：去除多余空行、统一换行符"""
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]  # 去除空行
        return "\n".join(lines)


class MarkdownLoader(BaseDocumentLoader):
    """
    Markdown 加载器
    
    Markdown 的特殊价值：标题层级（# ## ###）是天然的分块边界信号。
    我们这里先做基础加载，第3课分块时会利用这些标题信息。
    """

    def load(self, file_path: str) -> List[DocumentChunk]:
        filename = Path(file_path).name
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Markdown 作为一个整体加载，因为分块策略会智能处理标题
        pages = [LangChainDocument(
            page_content=content,
            metadata={
                "file_path": file_path,
                # 预留：后续可用正则提取所有标题，用于增强检索
                "has_headers": content.startswith("#"),
            }
        )]

        return self._make_chunks_from_pages(
            pages, filename, DocumentType.MD
        )


class TXTLoader(BaseDocumentLoader):
    """
    纯文本加载器（不依赖任何外部库）

    直接按行读取文本文件，按段落分割。
    适用于 .txt / .csv / .log 等纯文本格式。
    """

    name = "txt"
    description = "加载纯文本文件"

    def load(self, file_path: str) -> List[DocumentChunk]:
        filename = Path(file_path).name

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 按段落分割（空行分隔）
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        chunks = []
        for i, para in enumerate(paragraphs):
            chunks.append(DocumentChunk(
                content=para,
                metadata={
                    "source_file": filename,
                    "doc_type": "txt",
                    "paragraph_index": i,
                }
            ))

        return chunks


class UnstructuredLoader(BaseDocumentLoader):
    """
    万能加载器（基于 unstructured 库）
    
    支持：PDF, DOCX, PPTX, XLSX, HTML, TXT, CSV...
    优势：能识别表格、图片标题、列表等丰富元素类型
    劣势：依赖较多，首次安装和运行较慢
    
    安装命令：
        pip install unstructured[all-docs]
        # Windows 下可能需要额外安装 poppler（PDF）和 tesseract（OCR）
    """

    def load(self, file_path: str) -> List[DocumentChunk]:
        try:
            from unstructured.partition.auto import partition
        except ImportError:
            raise ImportError(
                "unstructured 未安装。请运行：pip install unstructured"
            )

        filename = Path(file_path).name
        suffix = Path(file_path).suffix.lower()

        # partition 是万能入口，根据文件扩展名自动选择解析器
        elements = partition(filename=file_path)

        # unstructured 的 Element 有 .text 和 .category（如 Title, NarrativeText, Table）
        # 我们把相同类型的连续元素合并为一个 chunk
        chunks = []
        current_text = []
        current_category = None

        for element in elements:
            category = str(type(element)).split(".")[-1].strip("'>")
            text = str(element)

            if not text.strip():
                continue

            # 标题单独成块（对检索很有价值）
            if "Title" in category:
                if current_text:
                    chunks.append(self._create_chunk(
                        current_text, current_category, filename
                    ))
                chunks.append(self._create_chunk(
                    [text], category, filename, is_title=True
                ))
                current_text = []
                current_category = None
            else:
                if current_category != category and current_text:
                    chunks.append(self._create_chunk(
                        current_text, current_category, filename
                    ))
                    current_text = []
                current_text.append(text)
                current_category = category

        # 处理最后一批
        if current_text:
            chunks.append(self._create_chunk(
                current_text, current_category, filename
            ))

        return chunks

    def _create_chunk(
        self,
        texts: List[str],
        category: Optional[str],
        filename: str,
        is_title: bool = False
    ) -> DocumentChunk:
        content = "\n".join(texts)
        return DocumentChunk(
            content=content,
            metadata={
                "source_file": filename,
                "element_category": category,
                "is_title": is_title,
            }
        )


# ═══════════════════════════════════════════════════════════
# 工厂类：统一入口
# ═══════════════════════════════════════════════════════════

class DocumentLoaderFactory:
    """
    文档加载器工厂
    
    使用方式：
        loader = DocumentLoaderFactory.get_loader("doc.pdf")
        chunks = loader.load("doc.pdf")
    """

    # 扩展名 → 加载器类的映射表
    _registry = {
        ".pdf": PDFLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        # 更多格式默认用 unstructured 处理
        ".docx": UnstructuredLoader,
        ".doc": UnstructuredLoader,
        ".pptx": UnstructuredLoader,
        ".ppt": UnstructuredLoader,
        ".xlsx": UnstructuredLoader,
        ".xls": UnstructuredLoader,
        ".html": UnstructuredLoader,
        ".htm": UnstructuredLoader,
        ".txt": TXTLoader,
        ".csv": UnstructuredLoader,
    }

    @classmethod
    def get_loader(cls, file_path: str) -> BaseDocumentLoader:
        """根据文件扩展名返回对应的加载器实例"""
        suffix = Path(file_path).suffix.lower()
        
        if suffix not in cls._registry:
            raise ValueError(f"不支持的文件格式: {suffix}。路径: {file_path}")
        
        loader_class = cls._registry[suffix]
        return loader_class()

    @classmethod
    def register(cls, extension: str, loader_class: type):
        """
        注册新的加载器（扩展用）
        
        示例：
            DocumentLoaderFactory.register(".epub", EPUBLoader)
        """
        cls._registry[extension.lower()] = loader_class


# ── 第2课小结 ──
#
# 你学到了：
# 1. 不同文档格式的解析陷阱（PDF扫描版、Word页眉页脚等）
# 2. 抽象基类 + 工厂模式：统一多格式加载的入口
# 3. PyMuPDF 用于 PDF，unstructured 作为万能兜底
# 4. 所有加载器最终都输出 List[DocumentChunk]，实现格式无关化
#
# 关键问题：加载后每页内容可能很长（几千字），不能直接用于向量检索。
# 下一节课我们将学习「分块策略」——如何把长文本切成语义完整的小块。
