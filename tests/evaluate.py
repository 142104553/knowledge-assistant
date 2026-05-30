"""
RAG 系统评测脚本

用法：
    # 先确保后端服务已启动
    python tests/evaluate.py --qa_file tests/qa_samples/protection_qa.json --output tests/results/protection_result.json
    python tests/evaluate.py --qa_file tests/qa_samples/all_qa.json --output tests/results/all_result.json

评测维度：
- correctness: 正确性（1-5分）
- completeness: 完整性（1-5分）
- overall_quality: 整体质量（1-5分）
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import requests
from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = "http://localhost:8000"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def call_rag(query: str) -> Dict[str, Any]:
    """调用 RAG 系统获取回答"""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json={
                "query": query,
                "session_id": f"eval_{int(time.time() * 1000)}",
                "top_k": 15,
                "enable_agent": False  # 纯 RAG 模式，排除 Agent 干扰
            },
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"answer": f"[错误] {e}", "sources": []}


def call_llm_judge(standard_answer: str, rag_answer: str, context: str, query: str) -> Dict[str, Any]:
    """调用 LLM 作为评判者，给回答打分"""
    try:
        import openai
    except ImportError:
        raise ImportError("请安装 openai: pip install openai")

    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")

    prompt = f"""你是一位严格的 RAG 系统质量评估专家。请对以下 RAG 系统的回答进行评分。

【用户问题】
{query}

【检索到的参考上下文】
{context[:1500]}

【标准答案（基于上下文的人工撰写答案）】
{standard_answer}

【RAG 系统实际生成的回答】
{rag_answer}

【评分要求】

请从以下三个维度评分（1-5分，5分最高）：

1. 正确性（correctness）：回答中的事实是否与标准答案一致？有无与上下文矛盾或编造的内容？
   - 5分：完全正确，所有事实与标准答案一致
   - 4分：基本正确，有轻微瑕疵
   - 3分：部分正确，有少量错误
   - 2分：错误较多
   - 1分：严重错误或与事实相反

2. 完整性（completeness）：回答是否涵盖了标准答案中的所有关键要点？有无重要遗漏？
   - 5分：完整覆盖所有关键要点
   - 4分：覆盖了大部分要点，有 minor 遗漏
   - 3分：覆盖了主要要点，有明显遗漏
   - 2分：只覆盖了一小部分
   - 1分：几乎未回答核心问题

3. 整体质量（overall_quality）：综合考虑正确性、完整性、表达清晰度、逻辑性。
   - 5分：优秀回答
   - 4分：良好回答
   - 3分：一般回答
   - 2分：较差回答
   - 1分：不合格回答

【输出格式】
请输出严格的 JSON 对象，不要包含 markdown 代码块：
{{
  "correctness": 分数,
  "completeness": 分数,
  "overall_quality": 分数,
  "analysis": "简要分析（50-100字）",
  "issues": ["发现的具体问题1", "问题2"]
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个严格、客观的评测专家。只基于提供的标准答案和上下文进行评判，不引入外部知识。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1500
    )

    raw = response.choices[0].message.content.strip()
    # 去掉可能的 markdown 代码块
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)

    # 尝试用正则提取 JSON 块（处理 LLM 在 JSON 前后加文字的情况）
    import re
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    try:
        result = json.loads(raw)
        # 确保字段存在
        for key in ["correctness", "completeness", "overall_quality"]:
            if key not in result:
                result[key] = 0
            result[key] = max(1, min(5, int(result[key])))
        return result
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"    [评分警告] JSON 解析失败，原始输出前200字: {raw[:200]}")
        return {
            "correctness": 0,
            "completeness": 0,
            "overall_quality": 0,
            "analysis": f"JSON 解析失败。原始输出: {raw[:200]}",
            "issues": [f"LLM 返回格式异常: {e}"]
        }


def evaluate_single(qa_item: Dict[str, Any]) -> Dict[str, Any]:
    """评测单个 QA 样本"""
    query = qa_item["query"]
    standard_answer = qa_item["answer"]
    context = qa_item.get("context", "")
    dimension = qa_item.get("dimension", "unknown")
    difficulty = qa_item.get("difficulty", "medium")

    print(f"\n  [评测] {query[:50]}...")

    # 1. 调用 RAG 获取回答
    rag_result = call_rag(query)
    rag_answer = rag_result.get("answer", "")
    print(f"  [RAG回答] {rag_answer[:300]}{'...' if len(rag_answer) > 300 else ''}")

    # 2. 调用 LLM 评判
    judge_result = call_llm_judge(standard_answer, rag_answer, context, query)

    return {
        "query": query,
        "dimension": dimension,
        "difficulty": difficulty,
        "standard_answer": standard_answer,
        "rag_answer": rag_answer,
        "scores": {
            "correctness": judge_result["correctness"],
            "completeness": judge_result["completeness"],
            "overall_quality": judge_result["overall_quality"]
        },
        "analysis": judge_result.get("analysis", ""),
        "issues": judge_result.get("issues", [])
    }


def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算统计指标"""
    if not results:
        return {}

    def avg(key: str) -> float:
        scores = [r["scores"][key] for r in results if r["scores"].get(key, 0) > 0]
        return sum(scores) / len(scores) if scores else 0

    stats = {
        "total_samples": len(results),
        "avg_correctness": round(avg("correctness"), 2),
        "avg_completeness": round(avg("completeness"), 2),
        "avg_overall_quality": round(avg("overall_quality"), 2),
        "dimension_breakdown": {},
        "difficulty_breakdown": {},
        "failed_samples": []
    }

    # 按维度统计
    dim_groups: Dict[str, List[int]] = {}
    diff_groups: Dict[str, List[int]] = {}

    for r in results:
        dim = r["dimension"]
        diff = r["difficulty"]
        overall = r["scores"]["overall_quality"]

        dim_groups.setdefault(dim, []).append(overall)
        diff_groups.setdefault(diff, []).append(overall)

        if overall <= 2:
            stats["failed_samples"].append({
                "query": r["query"],
                "dimension": dim,
                "overall_quality": overall,
                "issues": r.get("issues", [])
            })

    for dim, scores in dim_groups.items():
        stats["dimension_breakdown"][dim] = {
            "count": len(scores),
            "avg_overall": round(sum(scores) / len(scores), 2)
        }

    for diff, scores in diff_groups.items():
        stats["difficulty_breakdown"][diff] = {
            "count": len(scores),
            "avg_overall": round(sum(scores) / len(scores), 2)
        }

    return stats


def print_report(stats: Dict[str, Any]):
    """打印评测报告"""
    print("\n" + "=" * 60)
    print("📊 RAG 系统评测报告")
    print("=" * 60)
    print(f"总样本数: {stats['total_samples']}")
    print(f"平均正确性: {stats['avg_correctness']} / 5")
    print(f"平均完整性: {stats['avg_completeness']} / 5")
    print(f"平均整体质量: {stats['avg_overall_quality']} / 5")

    print("\n【按评测维度】")
    for dim, data in stats["dimension_breakdown"].items():
        print(f"  {dim:20s}: 样本{data['count']:3d} | 均分 {data['avg_overall']}")

    print("\n【按难度等级】")
    for diff, data in stats["difficulty_breakdown"].items():
        print(f"  {diff:10s}: 样本{data['count']:3d} | 均分 {data['avg_overall']}")

    if stats["failed_samples"]:
        print(f"\n【低质量样本 ({len(stats['failed_samples'])} 条)】")
        for item in stats["failed_samples"][:5]:
            print(f"  - [{item['dimension']}] {item['query'][:40]}... (均分 {item['overall_quality']})")
            for issue in item["issues"][:2]:
                print(f"      ⚠ {issue}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RAG 系统评测")
    parser.add_argument("--qa_file", required=True, help="QA 样本文件路径，如 tests/qa_samples/protection_qa.json")
    parser.add_argument("--output", default=None, help="评测结果输出路径")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条样本（用于快速测试）")
    args = parser.parse_args()

    # 加载 QA 样本
    qa_file = Path(args.qa_file)
    if not qa_file.exists():
        print(f"[错误] 文件不存在: {qa_file}")
        sys.exit(1)

    qa_list = json.loads(qa_file.read_text(encoding="utf-8"))
    if args.limit:
        qa_list = qa_list[:args.limit]

    print(f"[开始] 加载 {len(qa_list)} 条 QA 样本，来源: {qa_file}")
    print(f"[提示] 确保后端服务已启动: uvicorn app.api.main:app --port 8000")

    # 逐条评测
    results = []
    for i, qa in enumerate(qa_list, 1):
        print(f"\n[{i}/{len(qa_list)}]", end="")
        try:
            result = evaluate_single(qa)
            results.append(result)
        except Exception as e:
            print(f"[错误] 评测失败: {e}")
            results.append({
                "query": qa.get("query", ""),
                "dimension": qa.get("dimension", "unknown"),
                "scores": {"correctness": 0, "completeness": 0, "overall_quality": 0},
                "analysis": f"评测异常: {e}",
                "issues": [str(e)]
            })

    # 统计
    stats = compute_statistics(results)
    print_report(stats)

    # 保存结果
    output_data = {
        "eval_time": datetime.now().isoformat(),
        "qa_file": str(qa_file),
        "statistics": stats,
        "details": results
    }

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_DIR / f"{qa_file.stem}_result.json"

    output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] 详细结果已保存: {output_path}")


if __name__ == "__main__":
    main()
