---
name: project-context
description: 企业智能知识助手项目上下文引导（自动读取 PROJECT_CONTEXT.md）
---

## 项目上下文引导

**⚠️ 重要：本 Skill 为引导文件，最新项目上下文请以 `PROJECT_CONTEXT.md` 为准。**

### 自动读取指令

每次开始处理本项目任务时，**必须先阅读以下文件**获取最新项目全貌：

```
./PROJECT_CONTEXT.md
```

该文件包含：
- 技术栈与版本（Python 3.8 / MiMo-v2.5 / BGE / Chroma / SQLite）
- 完整目录结构与各模块职责
- 编码规范与约定
- 当前功能完成状态（✅ 已完成 / 🔄 进行中 / ⏳ 待开发）
- 已知问题与限制
- 按优先级排序的待办事项

### 如果 PROJECT_CONTEXT.md 不存在

基于项目实际代码扫描生成预填充模板，提示用户审阅保存后继续。

### 关键约束（摘要）

- **FastAPI 端点**：全部使用同步 `def`（非 `async def`）
- **文档身份**：`doc_id = MD5(file_bytes).hexdigest()`
- **Chunk 元数据**：每个 chunk 携带 `doc_id` + `source_file`
- **异常处理**：分层处理，不泄露敏感信息到客户端
- **数据库**：SQLite WAL 模式，`try/finally` 确保连接关闭
