---
name: project-context
description: 企业智能知识助手项目上下文，包含技术栈、目录结构、功能状态和编码规范
---

# 项目上下文：企业智能知识助手

## 1. 项目目标与范围

基于 RAG + LangChain + Agent 的私有领域知识库问答系统，面向电力行业技术文档提供智能检索与专业问答。

**边界**：
- ✅ 支持多格式文档解析、向量化存储、检索增强生成
- ✅ 支持文档生命周期管理（上传/删除/覆盖）
- ✅ 支持对话历史持久化
- ✅ 支持端到端评测框架
- ❌ 不涉及多模态（图像/视频）处理（v0.5 规划）
- ❌ 不涉及权限控制与审计日志（v1.0 规划）

## 2. 技术栈与版本

| 层级 | 技术 | 版本/说明 |
|:---|:---|:---|
| 语言 | Python | 3.8+ |
| LLM | MiMo-v2.5 | via OpenAI SDK, base_url 需含 /v1 |
| Embedding | BGE-small-zh-v1.5 | 512维, hf-mirror.com 下载 |
| 向量库 | Chroma | 本地持久化 ./chroma_db |
| 元数据 | SQLite | ./data/app.db |
| 后端 | FastAPI + Uvicorn | port 8000, 同步 def 端点 |
| 前端 | Streamlit | app/web/main.py |
| 框架 | LangChain | LCEL Chain, Agent Router |
| 文档解析 | PyMuPDF / 原生 / Unstructured | 工厂模式按扩展名路由 |
| 分块 | RecursiveCharacterTextSplitter | chunk_size=512, overlap=50-100 |
| 配置 | Pydantic Settings | .env + 环境变量, extra="ignore" |

## 3. 目录结构与职责

```
.
├── app/
│   ├── api/main.py          # FastAPI: chat, ingest, delete, stats, history
│   ├── web/main.py          # Streamlit: 批量上传、文档管理、聊天
│   └── core/config.py       # Pydantic Settings 配置中心
├── agent/
│   └── router.py            # Agent Router: 意图识别 + RAG Chain 调度
├── rag/
│   ├── chains/rag_chain.py  # RAGChain: retrieve → context builder → LLM
│   ├── retrievers/hybrid.py # HybridRetriever (Dense + BM25 RRF)
│   └── post_processors/reranker.py # CrossEncoderReranker
├── ingestion/
│   ├── loaders/factory.py   # Loader 工厂: PDF/MD/TXT/DOCX/PPTX/HTML
│   ├── loaders/txt_loader.py # 原生 TXT Loader（零依赖）
│   └── pipeline.py          # 端到端摄取: 解析→分块→Embedding→入库
├── embeddings/
│   └── factory.py           # Embedding 工厂: BGE 本地模型封装
├── vectorstore/
│   └── factory.py           # Chroma 实现 (delete_by_doc_id, get_all)
├── models/
│   ├── database.py          # SQLite: conversations + documents 表
│   └── document.py          # Pydantic 数据模型
├── tests/
│   ├── generate_corpus.py   # 生成 12 篇合成电力测试文档
│   ├── generate_qa.py       # 调用 LLM 生成带 dimension 标注的 QA
│   ├── evaluate.py          # 端到端评测: RAG→LLM评分→统计报告
│   ├── corpus/              # 4领域 × 3文档 测试语料
│   └── qa_samples/          # 25 条标注 QA (faithfulness/hallucination/...)
├── .env                     # 敏感配置, Git 忽略
├── .env.example             # 配置模板, 可提交
├── .gitignore               # 排除 .env, chroma_db/, data/, .idea/
├── PROJECT_CONTEXT.md       # 项目文档 (技术栈、数据管道、演进路线)
└── requirements.txt         # Python 依赖
```

## 4. 编码规范与约定

- **FastAPI 端点**：全部使用同步 `def`（非 `async def`），避免 BGE/Chroma/OpenAI 阻塞事件循环
- **文档身份**：`doc_id = MD5(file_bytes).hexdigest()`，相同内容自动去重/覆盖
- **Chunk 元数据**：每个 chunk 携带 `doc_id` + `source_file`，支持精确删除与溯源
- **错误处理**：API 异常打印 `traceback.print_exc()`，返回结构化错误响应
- **Commit 格式**：`feat/fix/refactor/docs(scope): 描述`

## 5. 当前功能状态

| 模块 | 状态 | 备注 |
|:---|:---|:---|
| 多格式文档解析 | ✅ 已完成 | PDF/TXT/MD/DOCX/PPTX/HTML |
| 文档生命周期管理 | ✅ 已完成 | 上传(覆盖)/删除, doc_id 追踪 |
| 本地 BGE Embedding | ✅ 已完成 | 512维, 中文优化 |
| Chroma 向量检索 | ✅ 已完成 | top_k=15, metadata filter |
| BM25 混合检索 | ✅ 已完成 | Dense + Sparse RRF 融合 |
| Cross-Encoder Reranker | ✅ 已完成 | BAAI/bge-reranker-base |
| MultiQuery 查询变体 | ✅ 已完成 | 3 变体并行检索 |
| 多文件上下文组装 | ✅ 已完成 | 每文件至少1 chunk, max_tokens=8000 |
| 对话历史持久化 | ✅ 已完成 | SQLite, Streamlit 自动加载 |
| Streamlit 前端 | ✅ 已完成 | 批量上传、进度条、文档管理、溯源 |
| FastAPI 后端 | ✅ 已完成 | 同步端点, 防阻塞 |
| 评估框架 | ✅ 已完成 | 语料→QA→评测 (4维度×3难度) |
| Agent Router 框架 | ✅ 已完成 | 基础意图识别, 可扩展 |

## 6. 已知问题与限制

1. **Embedding API 不可用**：MiMo provider 不支持 `/v1/embeddings`，必须使用本地 BGE
2. **BM25 语料库非实时更新**：新文档上传后需重启服务才能更新 BM25
3. **CLI 摄取元数据缺失**：`python -m ingestion.pipeline` 不入 SQLite `documents` 表
4. **Python 3.8 兼容**：f-string 反斜杠、posthog<3.0、Pydantic extra 字段已适配
5. **旧数据清理**：首次使用前建议清空 `./chroma_db` 和 `./data/app.db`
6. **评估 LLM JSON 解析偶发失败**：MiMo 返回格式不严格，已加正则提取 + 失败 0 分回退

## 7. 待办事项

| 优先级 | 事项 | 目标版本 |
|:---|:---|:---|
| P0 | Streaming 流式输出 | v0.4 |
| P0 | 结构化输出（Pydantic） | v0.4 |
| P1 | HyDE 假设文档嵌入 | v0.4 |
| P1 | Ragas 评估库接入 | v0.5 |
| P2 | LangSmith 监控 | v0.5 |
| P3 | 知识图谱 | v0.6 |
