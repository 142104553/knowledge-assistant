# VLM (视觉语言模型) 引入可行性计划

> 版本: v0.1
> 日期: 2026-06-06
> 状态: 设计阶段

---

## 1. 为什么要引入 VLM？

当前系统只能处理**纯文本文档**。但电力行业的实际文档（PDF/PPT/Word）中包含大量**非文本信息**：

| 文档类型 | 典型图片内容 |
|---------|-------------|
| 电力运维手册 | 接线图、设备照片、仪表读数截图 |
| 事故案例分析 | 故障现场照片、保护装置录波图、GIS 截图 |
| 技术规程 | 系统架构图、流程图、参数表格截图 |
| 培训课件 (PPT) | 示意图、对比图、动画截图 |

**当前痛点**：
- 用户问"保护装置的接线方式"→ 检索不到，因为接线图是图片，没提取文字
- 用户问"这个故障的录波图说明什么"→ 完全无法回答
- 上传 PDF 后，里面的图片被直接跳过，信息丢失 30%~50%

**VLM 能解决的问题**：
- 理解图片内容，生成文字描述
- 回答"图中显示了什么"
- 根据图片进行推理（如"从录波图判断故障类型"）

---

## 2. 三种技术方案对比

### 方案 A：图片描述生成 + 文本 RAG（推荐首选）

**思路**：
```
文档解析阶段：
  PDF/PPT → 提取文本 chunk（现有流程）
         → 提取图片 → VLM 生成描述 → 作为额外 chunk 存入向量库

查询阶段：
  用户文本查询 → 向量检索（可匹配到图片描述）→ LLM 生成回答
```

**优点**：
- 改动最小，复用现有 RAG 流程
- 不需要更换 Embedding 模型
- 检索逻辑完全不变
- 实现周期短（1~2 周）

**缺点**：
- VLM 调用成本高（每页 PPT 可能有 5~10 张图）
- 图片描述可能丢失细节
- 用户不能直接上传图片提问

**成本估算**（以 1000 页 PDF，每页 2 张图为例）：
- VLM API 调用：2000 次 × ¥0.05/次 = ¥100（一次性预处理成本）
- 额外存储：2000 个图片描述 chunk ≈ 10MB 向量数据

---

### 方案 B：多模态 Embedding（CLIP / BGE-VL）

**思路**：
```
索引阶段：
  文本 → BGE 编码 → 512D 向量
  图片 → CLIP/BGE-VL 编码 → 512D 向量（与文本同一空间）
  → 同一向量库存储

查询阶段：
  用户文本查询 → CLIP 编码 → 向量检索可同时命中文本和图片
```

**优点**：
- 检索更精准（跨模态语义匹配）
- 无需 VLM API 成本（本地模型推理）
- 用户文本查询可直接找到相关图片

**缺点**：
- 需要增加一个图片 Embedding 模型（显存占用 +1GB）
- CLIP 对中文电力专业术语理解较弱
- 需要维护两个 Embedding 管道
- 实现复杂度中等（2~3 周）

**成本估算**：
- CLIP 模型下载：~500MB
- 显存占用：+1GB（可与 BGE 共享 GPU）
- 推理速度：每张图 ~50ms（GPU）

---

### 方案 C：原生多模态 RAG（最完整）

**思路**：
```
索引阶段：
  文本 → BGE 编码 → 存 Milvus text_collection
  图片 → CLIP 编码 → 存 Milvus image_collection

检索阶段：
  用户查询 → 同时检索 text_collection + image_collection
         → RRF 融合结果
         → 结果包含：相关文本 + 相关图片原始文件

生成阶段：
  LLM/VLM 同时接收文本上下文 + 图片 base64
  → 生成"根据文本和下图可知..."类型的回答
```

**优点**：
- 最完整的多模态能力
- 可以回答"结合文字说明和图示解释..."
- 支持用户直接上传图片提问

**缺点**：
- 架构最复杂（需要修改 ingestion、retrieval、generation 三层）
- 需要 VLM API 支持（GPT-4V / Qwen-VL / InternVL）
- 图片传输增加带宽和延迟
- 实现周期长（4~6 周）

**成本估算**（按 1000 次问答/月）：
- VLM API：1000 次 × ¥0.15/次 = ¥150/月
- 图片存储：原始图片 ~500MB
- 带宽：每张图 ~200KB

---

## 3. 推荐路线图（渐进式实施）

> 原则：**先解决"有"的问题，再解决"好"的问题**

### Phase 1：图片描述生成（方案 A）— 2 周

**目标**：让系统能"看懂"文档中的图片，不遗漏信息

**改动点**：
1. `ingestion/loaders/` 增加图片提取能力
   - PyMuPDF: `page.get_images()` 提取 PDF 内嵌图片
   - python-pptx: 提取 PPT 幻灯片图片
   - 保存图片到 `./data/images/{doc_id}/{page}_{index}.png`
2. `ingestion/pipeline.py` 增加 VLM 描述生成步骤
   - 调用 VLM API（Qwen-VL / GPT-4V）生成图片描述
   - 将图片描述作为 `DocumentChunk` 存入向量库
   - metadata 标记 `chunk_type: "image_description"`, `image_path: "..."`
3. `rag/chains/rag_chain.py` 无需改动（复用现有流程）

**技术选型**：
- VLM 模型：Qwen2-VL-7B（阿里云 DashScope API，中文电力场景优化好）
- 备选：GPT-4V（OpenAI API，能力强但贵）

**Prompt 示例**：
```
你是一位电力系统专家。请详细描述这张图片的内容。
要求：
1. 识别图中的设备类型、接线方式、参数标注
2. 如果是录波图，描述波形特征和异常点
3. 如果是表格，提取所有数值参数
4. 用中文回答，200~500字
```

**验收标准**：
- 上传含图片的 PDF 后，向量库中同时存在文本 chunk 和图片描述 chunk
- 查询图片相关内容时，能检索到图片描述并生成合理回答

---

### Phase 2：多模态 Embedding（方案 B）— 3 周

**目标**：实现跨模态检索，文本查询可直接找到相关图片

**改动点**：
1. `embeddings/factory.py` 增加 `MultiModalEmbeddingClient`
   - 文本分支：BGE-small-zh-v1.5（现有）
   - 图片分支：BAAI/bge-visualized 或 openai/clip-vit-base-patch32
2. `ingestion/pipeline.py` 图片也生成向量
   - 图片 → CLIP 编码 → 存入 Milvus（与文本同一 collection）
   - metadata 增加 `chunk_type: "image"`
3. `vectorstore/milvus_store.py` 无需改动（已支持）

**技术选型**：
- 图片 Embedding: `BAAI/bge-visualized`（中文优化，与 BGE 语义空间兼容）
- 或 `openai/clip-vit-base-patch32`（社区生态好）

**验收标准**：
- 纯文本查询"保护装置接线图"能检索到相关图片
- 图片和文本在同一个向量空间，距离可比

---

### Phase 3：原生 VLM 回答（方案 C）— 4 周

**目标**：支持用户上传图片提问，回答中可引用原始图片

**改动点**：
1. `app/api/main.py` 新增 `/api/v1/chat/image` 端点
   - 接收：query + image_file（multipart）
   - 流程：图片 → VLM 理解 → 结合检索到的文本 → 生成回答
2. `models/document.py` 新增 `ImageQueryRequest`
3. `rag/chains/rag_chain.py` 新增 `multimodal_invoke()`
   - 检索文本上下文（现有流程）
   - 将图片 base64 + 文本上下文一起传给 VLM
4. `app/web/main.py` 前端支持图片上传

**技术选型**：
- VLM: Qwen2-VL-72B（能力强，支持多图理解）
- 或 InternVL2-Llama3-76B（开源，可本地部署）

**验收标准**：
- 用户上传故障录波图，问"这是什么故障类型"，系统能正确识别
- 回答中可显示"参考来源：xxx.pdf 第5页图示"

---

## 4. 技术架构演进

```
当前架构（纯文本）
====================
文档 → 文本提取 → BGE Embedding → Chroma/Milvus → 检索 → LLM

Phase 1（图片描述）
====================
文档 → 文本提取 ─┬─→ BGE Embedding ──→ Milvus ─┐
              └─→ 图片提取 → VLM描述 → BGE Embedding ─┘ → 检索 → LLM

Phase 2（多模态检索）
====================
文档 → 文本提取 ──→ BGE Embedding ─────┬─→ Milvus ─┐
              └─→ 图片提取 → CLIP Embedding ─┘          → 检索 → LLM

Phase 3（原生多模态 RAG）
=========================
文档 → 文本提取 ──→ BGE Embedding ─→ Milvus text_coll ─┬─→ RRF融合 → VLM
              └─→ 图片提取 → CLIP Embedding → Milvus image_coll ─┘
                                                                    ↑
用户上传图片 ──→ VLM 理解 ──────────────────────────────────────────┘
```

---

## 5. 关键改动清单

| 文件 | Phase 1 | Phase 2 | Phase 3 |
|------|---------|---------|---------|
| `ingestion/loaders/pdf_loader.py` | +图片提取 | - | - |
| `ingestion/loaders/pptx_loader.py` | +图片提取 | - | - |
| `ingestion/pipeline.py` | +VLM描述 | +CLIP编码 | - |
| `embeddings/factory.py` | - | +MultiModalClient | - |
| `rag/chains/rag_chain.py` | - | - | +multimodal_invoke |
| `app/api/main.py` | - | - | +/chat/image |
| `app/web/main.py` | - | - | +图片上传UI |
| `models/document.py` | +chunk_type | - | +ImageQueryRequest |

---

## 6. 依赖清单

```txt
# Phase 1: 图片提取 + VLM
PyMuPDF>=1.23.0        # 已有，升级以支持 get_images
python-pptx>=0.6.21    # 已有
requests>=2.31.0       # 已有（用于调用 VLM API）
# VLM API: DashScope (Qwen-VL) 或 OpenAI (GPT-4V)

# Phase 2: 多模态 Embedding
transformers>=4.40.0   # 加载 CLIP/BGE-VL
torchvision>=2.0.0     # 图片预处理
pillow>=10.0.0         # 图片格式转换
# 模型: BAAI/bge-visualized 或 openai/clip-vit-base-patch32

# Phase 3: 原生 VLM
dashscope>=1.20.0      # 阿里云 Qwen-VL API（可选）
# 或本地部署: lmdeploy / vLLM + InternVL2
```

---

## 7. 成本与风险评估

### 成本对比

| 阶段 | 一次性成本 | 月度运营成本 | 显存增加 |
|------|-----------|-------------|---------|
| Phase 1 | ¥100~500（VLM API） | ¥0（无增量） | 0 |
| Phase 2 | ¥0（开源模型） | ¥0 | +1GB GPU |
| Phase 3 | ¥0 | ¥150~500（VLM API） | +8~16GB（本地VLM）或 0（API）|

### 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| VLM 描述不准确 | 检索质量下降 | Phase 1 中增加人工审核抽样；Prompt 工程优化 |
| CLIP 对中文电力术语理解差 | 跨模态检索效果差 | 选用 BGE-VL（中文优化）；或 Phase 2 跳过，直接 Phase 3 |
| 图片数量过多导致 API 费用高 | 预算超支 | 限制图片尺寸（>200KB 才处理）；只处理前 N 张图/文档 |
| 显存不足 | 无法加载多模态模型 | 使用 API 方式（DashScope/OpenAI）；或量化模型（INT8） |

---

## 8. 结论与建议

### 立即可以做（Phase 1）
- 投入小（2 周），收益大（解决 50% 信息丢失问题）
- 推荐先用 **Qwen-VL-Plus（DashScope API）**，中文电力场景表现好
- 每页 PDF 只处理前 2 张最大的图，控制成本

### 暂缓评估（Phase 2 & 3）
- 等 Phase 1 上线后，观察用户实际 query 中涉及图片的比例
- 如果 <10%，Phase 2/3 优先级降低
- 如果 >30%，启动 Phase 2

### 最小可行产品（MVP）
```python
# 核心代码只需 ~50 行
import fitz  # PyMuPDF
from PIL import Image
import io

def extract_images_from_pdf(pdf_path, doc_id):
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            # 调用 VLM API 生成描述
            description = vlm_describe(image)
            # 存入向量库
            yield DocumentChunk(content=description, metadata={"doc_id": doc_id, "page": page_num, "type": "image"})
```

---

**下一步行动**：是否需要我直接实现 Phase 1 的 MVP（PDF 图片提取 + VLM 描述生成）？
