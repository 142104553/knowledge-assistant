# 企业智能知识助手 (Enterprise Knowledge Assistant)

基于 **RAG (检索增强生成) + LangChain + Agent** 架构构建的私有领域知识库问答系统。支持对企业内部文档、学术论文、产品手册等非结构化数据进行智能解析与向量化存储，并通过具备任务规划能力的 AI Agent 提供准确、可溯源的专业问答服务。

---

## 📌 项目目的

解决通用大语言模型在垂直领域的三大痛点：
- **知识幻觉**：基于私有知识库检索，所有回答均有据可查
- **知识滞后**：支持动态更新企业最新文档，无需重新训练模型
- **能力单一**：通过 Agent 架构，不仅能回答问题，还能执行摘要、对比、分析等复合任务

---

## 🏗️ 技术栈

| 层级 | 技术选型 | 说明 |
|:---|:---|:---|
| **应用层** | Streamlit / FastAPI | 提供 Web 交互界面与 RESTful API 服务 |
| **Agent 层** | LangChain Agents (ReAct / Plan-and-Execute) | 意图识别、任务规划、工具调度与多轮推理 |
| **RAG 层** | LangChain Retrieval Chain | 检索策略编排（多路召回、重排序、上下文压缩） |
| **LLM 层** | OpenAI API / Azure OpenAI / 本地大模型 (Ollama/vLLM) | 自然语言理解与生成 |
| **Embedding 层** | OpenAI Embedding / BGE / M3E | 文本语义向量化 |
| **向量数据库** | Milvus / Chroma / Qdrant | 高维向量存储与相似度检索 |
| **文档解析** | LangChain Document Loaders + Unstructured | 支持 PDF、Word、PPT、Markdown、HTML 等格式 |
| **分块策略** | RecursiveCharacterTextSplitter / Semantic Chunker | 智能文本切分，保留语义完整性 |
| **Orchestration** | LangGraph (可选) | 构建复杂的多节点工作流与状态机 |

---

## 🔄 数据流架构

系统数据流分为 **离线 ingestion 管道** 与 **在线查询管道** 两条主线：

### 1. 离线数据流（文档摄取与索引）

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  原始文档输入    │     │   文档解析引擎    │     │   文本分块处理   │
│ (PDF/DOCX/PPT/  │────▶│ (Unstructured /  │────▶│ (Recursive /    │
│  MD/HTML/CSV)   │     │  PyPDF / OCR)    │     │  Semantic Split)│
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│  向量数据库索引  │◄────│  Embedding 向量化 │◄─────────────┘
│ (Milvus/Chroma) │     │ (BGE / OpenAI    │
└─────────────────┘     │  Embedding)       │
                        └──────────────────┘
```

**流程说明：**
1. **文档加载**：通过 `UnstructuredFileLoader`、`PyPDFLoader` 等解析多格式文档，提取文本、表格及元数据（文件名、页码、章节）
2. **内容清洗**：去除页眉页脚、重复空格、无意义符号，保留文档层级结构
3. **智能分块**：采用递归字符切分（按段落/句子/单词优先级）或语义切分，控制 chunk_size（建议 512-1024 tokens）与重叠区间（overlap 50-200）
4. **向量化**：通过 Embedding 模型将文本块映射为高维向量
5. **索引存储**：将向量与原始文本、元数据一并写入向量数据库，建立高效近似最近邻（ANN）索引（如 HNSW）

### 2. 在线查询流（问答与 Agent 推理）

```
                              ┌─────────────────────────────────────┐
                              │            用户提问                 │
                              └──────────────┬──────────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────────────┐
                              │   Query Analyzer（查询分析）         │
                              │  • 意图识别（问答/摘要/对比）        │
                              │  • Query 重写与扩展（HyDE/Rewrite）  │
                              └──────────────┬──────────────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
        ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
        │   Agent Router    │   │   多路向量检索     │   │   工具/API 调用   │
        │  (ReAct 规划器)    │   │  • Dense Search   │   │  (如需要外部数据)  │
        │  • 任务分解        │   │  • Keyword Search │   └─────────┬─────────┘
        │  • 工具选择        │   │  • Rerank 重排序   │             │
        └─────────┬─────────┘   └─────────┬─────────┘             │
                  │                       │                       │
                  └───────────────┬───────┴───────────────────────┘
                                  ▼
                    ┌─────────────────────────────────────┐
                    │    Context Builder（上下文组装）      │
                    │  • 检索结果过滤与去重                 │
                    │  • 相关性阈值控制（Score Threshold）  │
                    │  • 上下文窗口压缩（Map-Reduce）       │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │         LLM 生成回答                 │
                    │  • 基于检索上下文进行推理             │
                    │  • 拒绝回答无依据问题                 │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │         回答后处理与输出              │
                    │  • 答案格式化（Markdown/JSON）        │
                    │  • 来源追溯（Source Citation）        │
                    │  • 多轮对话历史管理                   │
                    └─────────────────────────────────────┘
```

**流程说明：**
1. **查询分析**：对用户问题进行意图分类（ factual 问答 / 摘要总结 / 跨文档对比 / 多步推理），必要时进行 Query 重写或 HyDE（假设文档嵌入）扩展以提升检索质量
2. **Agent 路由**：LangChain Agent（基于 ReAct 或 Plan-and-Execute）根据意图决定执行策略：
   - 直接调用 RAG 检索链
   - 分解为多个子问题分别检索后综合
   - 调用外部工具（如计算器、数据库、搜索引擎）补充信息
3. **多路检索召回**：结合向量相似度检索（Dense Retrieval）与关键词检索（BM25/TF-IDF），通过 Reranker（如 Cross-Encoder）对混合结果重排序，提升召回准确率
4. **上下文组装**：根据 LLM 上下文窗口限制，对检索到的文档块进行过滤、去重与压缩，构建最优 prompt 上下文
5. **生成与溯源**：LLM 基于给定上下文生成回答，严格约束不编造知识；输出附带引用来源（文档名、页码、原文片段），确保可验证性
6. **对话管理**：维护多轮对话历史，支持上下文关联追问

---

## 📁 项目结构（推荐）

```
.
├── app/                          # 应用层
│   ├── api/                      # FastAPI 路由与接口定义
│   ├── web/                      # Streamlit 前端界面
│   └── core/                     # 应用配置与依赖注入
├── agent/                        # Agent 核心逻辑
│   ├── router.py                 # Agent 路由与意图识别
│   ├── tools/                    # 自定义工具集（检索工具、计算工具等）
│   └── prompts/                  # Agent 提示词模板
├── rag/                          # RAG 检索链
│   ├── retrievers/               # 检索器实现（向量、关键词、混合）
│   ├── chains/                   # LangChain LCEL 链定义
│   └── post_processors/          # 重排序、上下文压缩
├── ingestion/                    # 离线文档摄取管道
│   ├── loaders/                  # 各类文档加载器
│   ├── splitters/                # 分块策略实现
│   └── pipeline.py               # 端到端 ingestion 流程
├── embeddings/                   # Embedding 服务封装
│   └── factory.py
├── vectorstore/                  # 向量数据库封装
│   └── factory.py
├── models/                       # 数据模型（Pydantic）
├── config/                       # 配置文件（YAML/Env）
├── tests/                        # 单元测试与集成测试
├── docs/                         # 项目文档
├── scripts/                      # 工具脚本
├── README.md
└── requirements.txt
```

---

## 🚀 快速开始

### 环境要求
- Python >= 3.10
- 向量数据库（本地开发推荐 Chroma，生产推荐 Milvus/Qdrant）
- OpenAI API Key 或本地部署的 LLM 服务

### 安装依赖

```bash
# 克隆项目
git clone <repo-url>
cd enterprise-knowledge-assistant

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件：
```env
# LLM 配置
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o

# Embedding 配置
EMBEDDING_MODEL=text-embedding-3-small

# 向量数据库
VECTORSTORE_TYPE=chroma
CHROMA_PERSIST_DIR=./chroma_db

# 文档目录
DOCUMENTS_DIR=./data/documents
```

### 构建知识库索引

```bash
python -m ingestion.pipeline --input-dir ./data/documents
```

### 启动服务

**Web 界面（Streamlit）：**
```bash
streamlit run app/web/main.py
```

**API 服务（FastAPI）：**
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

---

## ✨ 核心特性

| 特性 | 描述 |
|:---|:---|
| **多格式支持** | 自动解析 PDF、Word、PPT、Excel、Markdown、HTML 等常见文档格式 |
| **混合检索** | Dense + Sparse 双路召回，结合 Cross-Encoder 重排序，兼顾召回率与准确率 |
| **Agent 规划** | 支持 ReAct、Plan-and-Execute 等多种 Agent 范式，自动分解复杂问题 |
| **来源可追溯** | 每个回答均附带原文引用，点击即可定位到出处文档及具体位置 |
| **多轮对话** | 自动维护对话上下文，支持指代消解与关联追问 |
| **拒答机制** | 当检索结果不足或无关时，明确告知用户"无法从现有知识库中找到答案" |
| **动态更新** | 支持增量文档导入与索引更新，无需全量重建 |

---

## 🛣️ 演进路线

- [ ] **v0.1** 基础 RAG 问答（单轮检索 + 生成）
- [ ] **v0.2** 引入 Agent 架构，支持多步推理与工具调用
- [ ] **v0.3** 混合检索与重排序优化
- [ ] **v0.4** 多轮对话与对话历史管理
- [ ] **v0.5** 接入 LangGraph 实现复杂工作流编排
- [ ] **v1.0** 权限控制、审计日志、企业级部署

---

## 📄 License

[MIT License](LICENSE)
