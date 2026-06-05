# 企业智能知识助手 (Enterprise Knowledge Assistant)

基于 **RAG (检索增强生成) + LangChain + Agent** 架构构建的私有领域知识库问答系统。支持对企业内部文档（电力行业技术规程、运维手册、事故案例等）进行智能解析、向量化存储与检索，并通过 LLM 生成准确、可溯源的专业问答服务。

> **当前版本**：v0.5 — 已实现 LangGraph Agent 状态机、三状态置信度判断（answerable/low_confidence/not_found）、工具调用（Calculator/DatabaseQuery）、AST 安全表达式解析、BM25 热更新、CORS 安全配置。

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
| **Agent 层** | LangChain Agents + LangGraph | 双架构 Agent Router（legacy if-else / LangGraph StateGraph）+ TOOL_CALL 工具调用 |
| **RAG 层** | LangChain LCEL Chain | MultiQuery → Hybrid Retrieval (BM25+Dense) → Cross-Encoder Rerank → Context Builder → LLM Generate |
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
- **配置热切换**：`.env` 中 `LLM_CLIENT_TYPE`（openai/langchain）和 `AGENT_ROUTER_TYPE`（legacy/langgraph）独立正交切换，零代码侵入回退到原始系统

---

## 📦 本地模型下载

本项目依赖以下 HuggingFace 模型，**首次运行前需下载到本地**（模型文件体积较大，不纳入 Git 管理）：

| 模型 | 用途 | 大小 | 下载命令 |
|:---|:---|:---|:---|
| `BAAI/bge-small-zh-v1.5` | Embedding（语义编码） | ~100MB | 首次运行时自动缓存到 `~/.cache/huggingface` |
| `BAAI/bge-reranker-base` | Cross-Encoder（精排） | ~1.1GB | **需手动下载到 `./models/bge-reranker-base/`** |

### 下载 Reranker 模型

```bash
# 方式一：通过 hf-mirror 镜像下载（推荐国内网络）
export HF_ENDPOINT=https://hf-mirror.com
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='BAAI/bge-reranker-base',
    local_dir='./models/bge-reranker-base',
    local_dir_use_symlinks=False
)
"

# 方式二：从其他已下载机器直接复制
# cp -r /path/to/bge-reranker-base ./models/
```

下载完成后，在 `.env` 中确认配置（已默认配置）：
```env
RERANKER_MODEL=BAAI/bge-reranker-base
RERANKER_LOCAL_PATH=./models/bge-reranker-base
```

> **注意**：若未下载模型，系统会自动降级为 `NoOpReranker`（无精排），检索质量会下降。

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
│ Agent Router    │  ← 意图识别：factual_qa / comparison / summarization /
│ 意图路由         │    multi_step / tool_call / chitchat
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MultiQuery      │  ← LLM 生成 3 个查询变体，覆盖不同关键词和语义角度
│ 查询扩展         │
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
│ Hybrid Retrieval│  ← Dense(Chroma) + Sparse(BM25) RRF 融合
│ 混合检索         │    top_k=15，where 过滤可选
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cross-Encoder   │  ← BAAI/bge-reranker-base 精排，分数 sigmoid 归一化到 [0,1]
│ Reranker        │    低于阈值触发 not_found / low_confidence
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
│ LLM Generate    │  ← MiMo-v2.5，三状态差异化 system prompt
│ 回答生成         │    answerable(正常) / low_confidence(谨慎) / not_found(拒答)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 输出 → 前端     │  ← 回答文本 + sources[] 溯源卡片 + answer_status 置信度
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
│   ├── router.py                # Legacy Agent Router：意图识别与工具调度（if-else 分支）
│   ├── langgraph_router.py      # LangGraph Agent Router：StateGraph 状态机 + ReAct 工具调用
│   └── tools/base.py            # 工具基类 + CalculatorTool + DatabaseQueryTool + ToolRegistry
├── rag/
│   ├── chains/rag_chain.py      # RAGChain：MultiQuery→检索→上下文组装→LLM生成 + 三状态置信度
│   ├── retrievers/hybrid.py     # HybridRetriever：Dense(Chroma) + Sparse(BM25) RRF 融合
│   └── post_processors/
│       └── reranker.py          # CrossEncoderReranker + ScoreThresholdFilter + NoOpReranker
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
- **Python 3.11+**（LangGraph 要求 Python ≥ 3.9，推荐 3.11）
- Windows / Linux / macOS
- 网络：可访问 `hf-mirror.com`（首次下载 BGE 模型）和 MiMo API
- **GPU 建议**：NVIDIA GPU 用于 Cross-Encoder 加速（CUDA 11.8，sm_61 及以上）

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
EMBEDDING_PROVIDER=bge
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DIMENSION=512
HF_ENDPOINT=https://hf-mirror.com

# 向量数据库
VECTORSTORE_PROVIDER=chroma
CHROMA_PERSIST_DIR=./chroma_db

# 元数据数据库（SQLite 自动初始化，无需手动创建）
# DATABASE_PATH=./data/app.db

# === Agent / LangChain 配置切换 ===
LLM_CLIENT_TYPE=langchain        # openai / langchain
AGENT_ROUTER_TYPE=langgraph      # legacy / langgraph
ENABLE_AGENT=true

# Reranker 模型（需手动下载到 ./models/bge-reranker-base/）
RERANKER_MODEL=BAAI/bge-reranker-base
RERANKER_LOCAL_PATH=./models/bge-reranker-base

# 回答置信度阈值（基于 Cross-Encoder 归一化分数）
ANSWER_STATUS_THRESHOLD_HIGH=0.6   # ≥0.6 → answerable
ANSWER_STATUS_THRESHOLD_LOW=0.3    # <0.3 → not_found；中间 → low_confidence

# CORS（生产环境应限制为具体域名）
CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
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

## ✨ 已实现功能

### v0.5 — 当前版本

| 功能模块 | 状态 | 说明 |
|:---|:---|:---|
| **LangGraph Agent Router** | ✅ | StateGraph 状态机：`analyze_intent → plan → route → [factual_qa\|tool_call\|...] → finalize` |
| **ReAct 工具调用** | ✅ | `calculator` + `database_query`，`TOOL:xxx\nARGS:{...}` 格式解析 |
| **三状态置信度判断** | ✅ | `answerable`(≥0.6) / `low_confidence`([0.3,0.6)) / `not_found`(<0.3)，差异化 system prompt |
| **Cross-Encoder 分数归一化** | ✅ | `torch.sigmoid` 将 logits 归一化到 [0,1]，统一阈值尺度 |
| **配置热切换** | ✅ | `LLM_CLIENT_TYPE`(openai/langchain) + `AGENT_ROUTER_TYPE`(legacy/langgraph) 正交组合，零侵入回退 |
| **BM25 热更新** | ✅ | 文档上传/删除后自动重建 BM25 语料库，无需重启 |
| **CORS 安全配置** | ✅ | 默认仅允许 localhost，生产环境通过 `.env` 配置白名单 |
| **CalculatorTool 安全加固** | ✅ | `eval()` → AST 安全解析器，完全阻断代码注入 |

### v0.4 — 检索质量优化

| 功能模块 | 状态 | 说明 |
|:---|:---|:---|
| **混合检索** | ✅ | BM25 + Dense 向量检索 RRF 融合（`HybridRetriever`） |
| **Cross-Encoder Reranker** | ✅ | `BAAI/bge-reranker-base` 精排，GPU 自动加速 |
| **MultiQuery 查询扩展** | ✅ | LLM 生成 3 个查询变体，并行检索合并去重 |
| **流式输出** | ✅ | FastAPI SSE + Streamlit `st.write_stream` |
| **结构化输出** | ✅ | Pydantic `EvaluationResult` + `response_format=json_object` |

### v0.3 — 基础能力

| 功能模块 | 状态 | 说明 |
|:---|:---|:---|
| **多格式文档解析** | ✅ | PDF(PyMuPDF)、TXT(原生)、MD(原生)、DOCX/PPTX/HTML(Unstructured) |
| **文档生命周期管理** | ✅ | 上传(自动去重/覆盖)、删除(向量+元数据双清)、文档列表查询 |
| **中文 Embedding** | ✅ | BGE-small-zh-v1.5 本地部署，512维 |
| **向量检索** | ✅ | Chroma 稠密检索，top_k=15，元数据过滤 |
| **多文件上下文组装** | ✅ | 按文件名分组，确保每个来源至少出现一次，max_tokens=8000 |
| **对话历史持久化** | ✅ | SQLite 存储，Streamlit 自动加载历史会话 |
| **前端界面** | ✅ | Streamlit：批量上传、文档管理、聊天溯源 |
| **RESTful API** | ✅ | FastAPI：chat/ingest/delete/stats/history |
| **评估框架** | ✅ | 自动生成语料 → 生成 QA → 端到端评测 |

---

## 🛣️ 演进路线

### 当前 (v0.5) — 已交付
- **LangGraph Agent 状态机**：`analyze_intent → plan → route → [factual_qa|comparison|summarization|multi_step|tool_call|chitchat] → finalize`
- **ReAct 工具调用**：`calculator`（AST 安全解析）+ `database_query`（mock 销售数据）
- **三状态置信度判断**：`answerable`(≥0.6) / `low_confidence`([0.3,0.6)) / `not_found`(<0.3)，reranker 分数 sigmoid 归一化
- **配置热切换**：`LLM_CLIENT_TYPE`(openai/langchain) + `AGENT_ROUTER_TYPE`(legacy/langgraph) 正交组合
- **BM25 热更新**：文档 upload/delete 后自动重建语料库
- **安全加固**：`eval()` → AST 解析器、CORS 白名单限制
- 完整的文档摄取 → 检索 → 生成 → 评估闭环
- 电力行业 4 领域测试语料与 25 条标注 QA

### v0.4 — 检索质量优化 ✅ 已完成
- [x] **混合检索**：BM25 + Dense 向量检索 RRF 融合（`HybridRetriever`）
- [x] **重排序器 (Reranker)**：`BAAI/bge-reranker-base` Cross-Encoder 精排，GPU 加速
- [x] **MultiQuery 查询扩展**：LLM 生成 3 个查询变体，并行检索合并去重
- [x] **流式输出**：FastAPI SSE + Streamlit `st.write_stream`
- [x] **结构化输出**：Pydantic `EvaluationResult` + `response_format=json_object`

### v0.6 — 检索与 Agent 增强
- [ ] **向量库升级（Milvus 原生混合检索）**：将 Dense + Sparse(BM25) 统一存入 Milvus，替代当前 Chroma + 外挂 `rank_bm25` 架构，实现插入即生效、无需重启、支持百万级规模
- [ ] **查询重写 (Query Rewrite)**：基于 LLM 的问题扩展与澄清，提升检索相关性（MultiQuery 已覆盖扩展部分，澄清式追问待补充）
- [ ] **HyDE (假设文档嵌入)**：用 LLM 生成伪答案再 Embedding，改善短查询检索效果
- [ ] **Agent 多步推理**：多轮 ReAct 循环（观察→思考→行动），支持"先查 A 再查 B 最后对比"类复合任务（当前为单轮工具调用）

### 远期 (v0.7-v1.0) — 生产级能力
- [x] **工具扩展**：`calculator` + `database_query` 已接入（🟡 待扩展更多实际业务工具）
- [ ] **多模态文档**：图像 OCR（表格、接线图）、PDF 内嵌图片解析
- [ ] **对话记忆压缩**：长对话历史自动摘要，避免上下文窗口溢出
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

1. **Embedding API 不可用**：当前 LLM provider（MiMo）不支持 `/v1/embeddings`，必须使用本地 BGE 模型
2. **CLI 摄取元数据缺失**：`python -m ingestion.pipeline` 不入 SQLite `documents` 表，前端列表不可见
3. **首次清理**：旧版 `chroma_db` 中的 chunk 可能缺少 `doc_id`，首次使用前建议清空 `./chroma_db` 和 `./data/app.db`
4. **Agent 流式输出**：Agent 模式（LangGraph）暂不支持 SSE 流式，fallback 到非流式一次性返回
5. **RAGAS 兼容性**：MiMo API 不支持 `n` 参数，`tests/ragas_eval.py` 标记为实验性脚本，未纳入正式评测
6. **单轮工具调用**：当前 `tool_call_node` 为单轮 ReAct（LLM → 工具 → 整合结果），多轮循环（观察→思考→行动）待扩展

---

## 📄 License

[MIT License](LICENSE)
