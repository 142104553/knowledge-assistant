# 企业智能知识助手 (Enterprise Knowledge Assistant)

基于 **RAG (检索增强生成) + LangChain + Agent** 架构构建的私有领域知识库问答系统。支持对企业内部文档（电力行业技术规程、运维手册、事故案例等）进行智能解析、向量化存储与检索，并通过 LLM 生成准确、可溯源的专业问答服务。

> **当前版本**：v0.3 — 已实现完整的文档生命周期管理、多领域 RAG 检索、对话持久化与评估框架。

---

## 📌 项目定位

解决通用大语言模型在垂直领域（电力系统调度、保护、配电、设备运维）的三大痛点：

- **知识幻觉**：所有回答均基于检索到的私有文档上下文，附带原文溯源
- **知识滞后**：支持动态上传/删除/覆盖文档，无需重新训练模型
- **领域专业度**：针对电力行业技术文档优化分块与检索策略，支持多文件交叉分析

---

## 🏗️ 技术栈

| 层级 | 技术选型 | 实际配置 |
|:---|:---|:---|
| **应用层** | Streamlit + FastAPI | Streamlit 前端 (`app/web/main.py`) + FastAPI/Uvicorn 后端 (`app/api/main.py:8000`) |
| **Agent 层** | LangChain Agents | Agent Router 意图识别 + RAG Chain 调用（ReAct 模式基础框架） |
| **RAG 层** | LangChain LCEL Chain | Dense Retrieval → Rerank → Context Builder → LLM Generate |
| **LLM 层** | OpenAI SDK (兼容接口) | MiMo-v2.5 (`https://token-plan-cn.xiaomimimo.com/v1`) |
| **Embedding** | HuggingFace `sentence-transformers` | BGE-small-zh-v1.5 (512维)，通过 `hf-mirror.com` 下载 |
| **向量数据库** | Chroma | 本地持久化 (`./chroma_db`)，元数据过滤 (`doc_id`) |
| **元数据数据库** | SQLite | 文档注册表 + 对话历史 (`./data/app.db`) |
| **文档解析** | 多 Loader 工厂 | PyMuPDF(PDF)、原生 Markdown、原生 TXT、Unstructured(DOCX/PPTX/HTML) |
| **分块策略** | `RecursiveCharacterTextSplitter` | chunk_size ~512 tokens，overlap 50-100 |
| **配置管理** | Pydantic Settings | 集中管理 `.env` + 环境变量，支持 `extra = "ignore"` |

### 关键技术决策

- **同步 FastAPI 端点**：所有 endpoint 使用 `def` 而非 `async def`，避免同步 BGE Embedding / Chroma 查询 / OpenAI LLM 调用阻塞 asyncio 事件循环
- **文档身份追踪**：`doc_id = MD5(file_bytes)`，相同文件内容自动去重/覆盖
- **Chunk 级元数据注入**：每个文本块都携带 `doc_id` + `source_file`，支持精确删除与溯源
- **中文 Embedding 本地部署**：BGE-small-zh-v1.5 针对中文语义优化，无需依赖外部 Embedding API

---

## 🔄 数据管道

系统由两条核心管道构成：**离线文档摄取管道** 与 **在线查询管道**。

### 1. 离线摄取管道 (Ingestion Pipeline)

```
用户上传文件 (API/CLI/前端)
         │
         ▼
┌─────────────────┐
│  Loader Factory │  ← 根据扩展名路由：PDF→PyMuPDF, TXT→原生, MD→原生,
│  文档解析        │     DOCX/PPTX/HTML→Unstructured
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Splitter   │  ← RecursiveCharacterTextSplitter
│ 智能分块         │    chunk_size=512, overlap=50-100
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  doc_id 生成    │────▶│  doc_id 覆盖检查 │  ← 相同 MD5 → 先删旧向量再入库
│  MD5(文件内容)  │     │  (Chroma + SQLite)
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ BGE Embedding   │  ← 批量编码 (batch_size=100)
│ 文本向量化       │    512维稠密向量，中文语义
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Chroma 向量库   │     │ SQLite 文档表   │
│ 存储向量+元数据  │     │ 存储文档元信息   │
│ (doc_id, source)│     │ (doc_id, name,  │
└─────────────────┘     │  size, time)    │
                        └─────────────────┘
```

**关键流程说明**：
1. **多格式解析**：工厂模式根据文件扩展名自动选择 Loader，TXT 走零依赖原生解析，避免 Unstructured 未安装时解析失败
2. **doc_id 生成**：`hashlib.md5(file_bytes).hexdigest()`，相同内容始终映射到同一 ID
3. **覆盖逻辑**：上传同名/同内容文件时，先通过 `where={"doc_id": ...}` 删除 Chroma 中旧 chunk，再写入新向量，同时 SQLite `INSERT OR REPLACE` 更新元数据
4. **批量编码**：Embedding 分批处理（每批100条），减少显存峰值

### 2. 在线查询管道 (Query Pipeline)

```
用户提问 (Streamlit / API)
         │
         ▼
┌─────────────────┐
│ Agent Router    │  ← 意图识别：当前默认路由至 RAG Chain
│ 意图路由         │    (可扩展：摘要、对比、计算等)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query Embedding│  ← BGE 编码用户问题 → 512维查询向量
│  问题向量化      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Chroma 相似检索 │  ← top_k=15 (原5→15)，余弦相似度
│ Dense Search    │    where 过滤可选
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Context Builder │  ← 按 source_file 分组，确保每个文件至少1个chunk
│ 上下文组装       │    剩余槽位按相似度分数填充，max_tokens=8000
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Generate    │  ← MiMo-v2.5，system prompt 强制要求多文件综合分析
│ 回答生成         │    附带来源引用（文档名+页码）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 输出 → 前端     │  ← 回答文本 + sources[] 溯源卡片
│ 溯源 + 持久化   │    SQLite 保存对话历史
└─────────────────┘
```

**关键流程说明**：
1. **扩大检索窗口**：`top_k=15`（原5），`max_context_tokens=8000`（原4000），确保多文档场景下信息不遗漏
2. **多文件均衡**：Context Builder 使用 `defaultdict` 按文件名分组，round-robin 保证每个来源文件至少出现一次，再按分数填充剩余位置
3. **同步执行**：FastAPI endpoint 使用 `def`（线程池执行），避免 async 事件循环被同步 LLM/Embedding 调用阻塞
4. **对话历史**：每次问答自动保存到 SQLite，Streamlit 侧边栏自动加载历史会话列表

---

## 📁 项目结构

```
.
├── app/
│   ├── api/main.py              # FastAPI 后端：chat, ingest, delete, stats, history
│   ├── web/main.py              # Streamlit 前端：批量上传、文档管理、聊天界面
│   └── core/config.py           # Pydantic Settings：LLM/Embedding/向量库配置
├── agent/
│   └── router.py                # Agent Router：意图识别与工具调度（基础框架）
├── rag/
│   ├── chains/rag_chain.py      # RAGChain：检索→上下文组装→LLM生成
│   └── retrievers/              # 检索器（Dense + Hybrid 扩展位）
├── ingestion/
│   ├── loaders/factory.py       # Loader 工厂：PDF/MD/TXT/DOCX/PPTX/HTML
│   ├── loaders/txt_loader.py    # 原生 TXT Loader（零依赖）
│   └── pipeline.py              # 端到端摄取：解析→分块→Embedding→入库
├── embeddings/
│   └── factory.py               # Embedding 工厂：BGE 本地模型封装
├── vectorstore/
│   └── factory.py               # 向量库工厂：Chroma 实现（delete_by_doc_id）
├── models/
│   └── database.py              # SQLite：conversations 表 + documents 表
├── tests/
│   ├── generate_corpus.py       # 生成 12 篇合成电力行业测试文档
│   ├── generate_qa.py           # 调用 LLM 生成带 dimension 标注的 QA 样本
│   ├── evaluate.py              # 端到端评测：RAG回答 → LLM评分 → 统计报告
│   ├── corpus/                  # 测试语料（4领域 × 3文档）
│   └── qa_samples/              # QA 样本（faithfulness / hallucination / ...）
├── chroma_db/                   # Chroma 向量数据持久化目录
├── data/app.db                  # SQLite 元数据与对话历史
└── requirements.txt             # Python 依赖
```

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Windows / Linux / macOS
- 网络：可访问 `hf-mirror.com`（首次下载 BGE 模型）和 MiMo API

### 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 安装依赖（使用国内镜像，HTTP trusted-host）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 配置环境变量

创建 `.env` 文件：

```env
# LLM 配置（MiMo-v2.5，base_url 必须包含 /v1）
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
LLM_MODEL=MiMo-v2.5

# Embedding 配置（本地 BGE，首次自动下载）
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DIMENSION=512
HF_ENDPOINT=https://hf-mirror.com

# 向量数据库
VECTORSTORE_TYPE=chroma
CHROMA_PERSIST_DIR=./chroma_db

# 元数据数据库
DATABASE_PATH=./data/app.db
```

### 初始化与启动

```bash
# 1. 首次启动：初始化数据库
python -c "from models.database import init_db; init_db()"

# 2. 启动后端 API
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. 启动前端（新终端）
streamlit run app/web/main.py
```

### 文档摄取（三种方式）

**方式一：前端批量上传**
打开 Streamlit 界面，侧边栏选择文件（支持多选），自动解析入库。

**方式二：API 上传**
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@your_document.pdf"
```

**方式三：CLI 命令行**
```bash
python -m ingestion.pipeline --input-path ./data/documents/
```
> 注：CLI 方式会写入向量库但**不写入 SQLite 文档表**，前端文档列表不可见。

---

## ✨ 已实现功能 (v0.3)

| 功能模块 | 状态 | 说明 |
|:---|:---|:---|
| **多格式文档解析** | ✅ | PDF(PyMuPDF)、TXT(原生)、MD(原生)、DOCX/PPTX/HTML(Unstructured) |
| **文档生命周期管理** | ✅ | 上传(自动去重/覆盖)、删除(向量+元数据双清)、文档列表查询 |
| **中文 Embedding** | ✅ | BGE-small-zh-v1.5 本地部署，512维，hf-mirror 镜像下载 |
| **向量检索** | ✅ | Chroma 稠密检索，top_k=15，元数据过滤 |
| **多文件上下文组装** | ✅ | 按文件名分组，确保每个来源至少出现一次，max_tokens=8000 |
| **对话历史持久化** | ✅ | SQLite 存储，Streamlit 自动加载历史会话 |
| **前端界面** | ✅ | Streamlit：批量上传（进度条）、文档管理（删除按钮）、聊天（溯源卡片） |
| **RESTful API** | ✅ | FastAPI：chat/ingest/delete/stats/history，同步 def 防阻塞 |
| **评估框架** | ✅ | 自动生成语料 → 生成 QA → 端到端评测（4维度 × 3难度） |
| **Agent 路由框架** | ✅ | 基础意图识别与 RAG Chain 调度，可扩展更多工具 |

---

## 🛣️ 演进路线

### 当前 (v0.3) — 已交付
- 完整的文档摄取 → 检索 → 生成 → 评估闭环
- 电力行业 4 领域（调度/保护/配电/设备）测试语料与 25 条标注 QA
- 文档生命周期一致性（API/前端 upload/delete/overwrite）

### 近期 (v0.4) — 检索质量优化
- [ ] **混合检索**：接入 BM25 关键词检索，与 Dense 向量检索融合（HybridRetriever）
- [ ] **重排序器 (Reranker)**：Cross-Encoder 对多路召回结果精排
- [ ] **查询重写 (Query Rewrite)**：基于 LLM 的问题扩展与澄清，提升检索相关性
- [ ] **HyDE (假设文档嵌入)**：用 LLM 生成伪答案再 Embedding，改善短查询检索效果

### 中期 (v0.5) — Agent 与多模态
- [ ] **Agent 多步推理**：ReAct / Plan-and-Execute 完整实现，支持"先查 A 再查 B 最后对比"类复合任务
- [ ] **工具扩展**：接入计算器、数据库查询、外部 API 等自定义 Tool
- [ ] **多模态文档**：图像 OCR（表格、接线图）、PDF 内嵌图片解析
- [ ] **对话记忆压缩**：长对话历史自动摘要，避免上下文窗口溢出

### 远期 (v0.6-v1.0) — 生产级能力
- [ ] **权限控制**：基于用户/角色的文档访问隔离（同一份向量库，不同可见范围）
- [ ] **审计日志**：完整记录问答内容、检索来源、模型参数，支持合规审查
- [ ] **增量索引**：监听文档目录变化，自动检测新增/修改/删除并同步更新索引
- [ ] **多模型切换**：支持同时配置多个 LLM（MiMo/ChatGPT/本地模型），按场景路由
- [ ] **容器化部署**：Docker + Docker Compose，支持 K8s 水平扩展
- [ ] **实时监控**：检索延迟、LLM 首字延迟、召回率、用户满意度 Dashboard

---

## 🧪 评测使用

```bash
# 1. 生成测试语料（12篇合成电力文档）
python tests/generate_corpus.py

# 2. 生成 QA 样本（4维度 × 3难度）
python tests/generate_qa.py --domain protection --n 5

# 3. 运行端到端评测（需后端已启动）
python tests/evaluate.py --qa_file tests/qa_samples/all_qa.json --output tests/results/all_result.json
```

评测报告输出：
- 平均正确性 / 完整性 / 整体质量（1-5分）
- 按维度（faithfulness / hallucination / noise_sensitivity / context_utilization）统计
- 按难度（easy / medium / hard）统计
- 失败样本明细与原始 RAG 回答

---

## ⚠️ 已知限制

1. **Embedding API 不可用**：当前 provider（MiMo）不支持 `/v1/embeddings`，必须使用本地 BGE 模型
2. **BM25 未启用**：`HybridRetriever` 仅 Dense 检索生效，BM25 语料库待初始化
3. **CLI 摄取元数据缺失**：`python -m ingestion.pipeline` 不入 SQLite `documents` 表，前端列表不可见
4. **首次清理**：旧版 `chroma_db` 中的 chunk 可能缺少 `doc_id`，首次使用前建议清空 `./chroma_db` 和 `./data/app.db`
5. **Python 3.8 兼容**：f-string 反斜杠、posthog 版本、Pydantic extra 字段等已做适配

---

## 📄 License

[MIT License](LICENSE)
