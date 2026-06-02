"""
============================================================================
⚠️  EXPERIMENTAL / 实验性脚本 — 不推荐用于生产评测
============================================================================

已知限制（在 Python 3.8 + MiMo API + 中文文档环境下已验证无法正常工作）：

1. MiMo API 不支持 RAGAS 内部的 'n' 参数 → 调用 LLM 时返回 400 BadRequest
2. RAGAS 0.1.x 的 faithfulness 指标无法从中文答案中提取 statement → 全返回 N/A
3. 评测速度极慢：单条样本 × 单个指标 ≈ 17~80 秒
4. RAGAS 0.2.x 要求 Python 3.9+，本项目锁定 Python 3.8 无法升级

✅ 正式评测请使用：tests/evaluate.py（LLM-as-a-Judge，稳定、可控、速度快）

本脚本仅作为技术储备保留，如需使用 RAGAS，请满足以下条件之一：
- 切换至支持完整 OpenAI API（含 'n' 参数）的 LLM 提供商
- 升级至 Python 3.9+ 并使用 ragas 0.2.x（但需自行验证中文支持）
============================================================================

RAGAS 独立评测脚本（兼容 Python 3.8 + ragas 0.1.x）

用法：
    # 快速验证（默认3条）
    python tests/ragas_eval.py

    # 全量评测
    python tests/ragas_eval.py --limit 25

    # 指定指标
    python tests/ragas_eval.py --metrics faithfulness,answer_relevancy

    # 指定输出路径
    python tests/ragas_eval.py --output tests/results/ragas_full.json

注意事项：
    - 需要后端服务已启动（默认 http://localhost:8000）
    - RAGAS 每个指标会多次调用 LLM，评测耗时较长
    - 建议先用 --limit 3 验证，再跑全量
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

# RAGAS
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_utilization,
        answer_correctness,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError as e:
    print(f"[ERROR] RAGAS import failed: {e}")
    print("[HINT] pip install 'ragas==0.1.21' (Python 3.8 compatible)")
    sys.exit(1)

# 强制 HuggingFace 离线模式（避免网络请求）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# LangChain 包装器（优先 langchain_openai/langchain_community）
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


API_BASE_URL = "http://localhost:8000"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def call_rag(query: str) -> Dict[str, Any]:
    """
    调用后端 RAG API，获取回答和检索上下文

    Returns:
        {
            "answer": str,
            "sources": [{"content": str, "metadata": {}, "score": float}, ...]
        }
    """
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json={
                "query": query,
                "session_id": f"ragas_{int(time.time() * 1000)}",
                "top_k": 15,
                "enable_agent": False  # 纯 RAG 模式，排除 Agent 干扰
            },
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "answer": data.get("answer", ""),
            "sources": data.get("sources", [])
        }
    except Exception as e:
        print(f"    [ERROR] RAG call failed: {e}")
        return {"answer": f"[ERROR] {e}", "sources": []}


def build_ragas_dataset(qa_items: List[Dict[str, Any]], limit: int = None) -> Dataset:
    """
    将 QA 样本转换为 RAGAS 数据集格式

    RAGAS 需要的字段：
        question:       List[str]           用户问题
        answer:         List[str]           RAG 生成的答案
        contexts:       List[List[str]]     每个问题对应的检索上下文列表
        ground_truth:   List[str]           （可选）人工标准答案
    """
    if limit:
        qa_items = qa_items[:limit]

    questions = []
    answers = []
    contexts = []          # list of lists
    ground_truths = []

    total = len(qa_items)
    for i, item in enumerate(qa_items, 1):
        query = item.get("query", "")
        standard_answer = item.get("answer", "")

        print(f"\n[{i}/{total}] RAG: {query[:60]}{'...' if len(query) > 60 else ''}")

        rag_result = call_rag(query)
        answer = rag_result["answer"]
        sources = rag_result["sources"]

        # 提取上下文文本列表（过滤空内容）
        source_texts = [
            s.get("content", "")
            for s in sources
            if s.get("content")
        ]

        preview = answer[:100] + "..." if len(answer) > 100 else answer
        print(f"    answer: {preview}")
        print(f"    sources: {len(source_texts)} chunks")

        questions.append(query)
        answers.append(answer)
        contexts.append(source_texts)
        ground_truths.append(standard_answer)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })


def get_ragas_metrics(metric_names_str: str) -> List[Any]:
    """根据逗号分隔的指标名返回 RAGAS 指标对象列表"""
    metric_map = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "context_utilization": context_utilization,
        "answer_correctness": answer_correctness,
    }

    metrics = []
    for name in metric_names_str.split(","):
        name = name.strip().lower()
        if name in metric_map:
            metrics.append(metric_map[name])
        else:
            print(f"[WARN] Unknown metric '{name}', skipping. Available: {list(metric_map.keys())}")

    return metrics


def print_report(result, output_path: str = None):
    """打印评测报告"""
    print("\n" + "=" * 60)
    print("[REPORT] RAGAS Evaluation Report")
    print("=" * 60)

    # ragas 0.1.x 返回 Result 对象，先转 pandas DataFrame
    result_df = result.to_pandas()

    # 只处理数值类型的指标列（跳过原始文本列）
    import numbers
    for col in result_df.columns:
        scores = result_df[col].tolist()
        # 过滤非数值、None 和 NaN
        valid_scores = [
            s for s in scores
            if isinstance(s, numbers.Number) and s == s  # 排除 NaN
        ]
        if valid_scores:
            avg = sum(valid_scores) / len(valid_scores)
            print(f"  {col:25s}: avg={avg:.4f}  (valid={len(valid_scores)}/{len(scores)})")
        else:
            print(f"  {col:25s}: N/A (no valid scores)")

    if output_path:
        result_df.to_json(output_path, orient="records", force_ascii=False, indent=2)
        print(f"\n[OK] Detailed results saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RAGAS RAG Evaluation")
    parser.add_argument(
        "--qa_file",
        default="tests/qa_samples/all_qa.json",
        help="QA 样本文件路径"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="结果输出 JSON 路径（默认自动生成）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="只评测前 N 条样本（默认 3，用于快速验证）"
    )
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,context_precision,context_recall,context_utilization",
        help="逗号分隔的指标列表"
    )
    args = parser.parse_args()

    # 加载 QA 样本
    qa_file = Path(args.qa_file)
    if not qa_file.exists():
        print(f"[ERROR] File not found: {qa_file}")
        sys.exit(1)

    with open(qa_file, "r", encoding="utf-8") as f:
        qa_list = json.load(f)

    print(f"[INFO] Loaded {len(qa_list)} QA samples from {qa_file}")
    print(f"[INFO] Backend API: {API_BASE_URL}")
    print(f"[INFO] Metrics: {args.metrics}")
    print(f"[INFO] Limit: {args.limit if args.limit else 'all'}")

    # Phase 1: 调用后端 RAG 收集 answer + contexts
    print("\n" + "=" * 60)
    print("Phase 1: Collecting RAG answers and contexts...")
    print("=" * 60)

    dataset = build_ragas_dataset(qa_list, limit=args.limit)

    # Phase 2: 初始化 RAGAS 所需的 LLM 和 Embedding
    print("\n[INFO] Initializing RAGAS LLM (MiMo) and Embedding (BGE)...")

    # LLM: MiMo via OpenAI-compatible API
    chat_model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "mimo-v2.5"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.3,
        max_retries=2,
        request_timeout=60,
    )
    ragas_llm = LangchainLLMWrapper(chat_model)

    # Embedding: BGE-small-zh-v1.5（本地模型，强制离线）
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu", "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True}
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # 选择指标
    metrics = get_ragas_metrics(args.metrics)
    if not metrics:
        print("[ERROR] No valid metrics selected")
        sys.exit(1)

    metric_names = [getattr(m, "name", str(m)) for m in metrics]
    print(f"[INFO] Running {len(metrics)} metrics: {metric_names}")

    # Phase 3: 运行 RAGAS 评估
    print("\n" + "=" * 60)
    print("Phase 2: Running RAGAS evaluation...")
    print("=" * 60)
    print("(Each metric calls LLM multiple times per sample, please wait)")

    start_time = time.time()
    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False  # 单条失败不中断整体评估
        )
    except Exception as e:
        print(f"\n[ERROR] RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = int((time.time() - start_time) * 1000)

    # 输出报告
    output_path = args.output
    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"ragas_eval_{ts}.json"

    print_report(result, output_path=str(output_path))
    print(f"\n[INFO] Evaluation completed in {elapsed}ms")


if __name__ == "__main__":
    main()
