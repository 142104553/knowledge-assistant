"""
生成 RAG 评测问答样本

用法：
    python tests/generate_qa.py --domain protection --n 10
    python tests/generate_qa.py --domain all --n 5

输出：tests/qa_samples/{domain}_qa.json
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 加载 .env 中的 API 配置
from dotenv import load_dotenv
load_dotenv()

CORPUS_DIR = Path(__file__).parent / "corpus"
OUTPUT_DIR = Path(__file__).parent / "qa_samples"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_corpus(domain: str) -> str:
    """加载某个领域的所有语料文本"""
    domain_dir = CORPUS_DIR / domain
    if not domain_dir.exists():
        raise ValueError(f"领域不存在: {domain}")

    texts = []
    for file_path in sorted(domain_dir.glob("*.txt")):
        texts.append(f"=== {file_path.name} ===\n{file_path.read_text(encoding='utf-8')}")

    return "\n\n".join(texts)


def call_llm(prompt: str, model: str = None) -> str:
    """调用 LLM API 生成文本"""
    try:
        import openai
    except ImportError:
        raise ImportError("请安装 openai: pip install openai")

    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个专业的电力系统知识库评测专家，擅长生成高质量的 RAG 评测问答样本。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=4000
    )
    return response.choices[0].message.content


def generate_qa_prompt(corpus_text: str, domain: str, n: int) -> str:
    """
    构造生成 QA 的 prompt

    要求覆盖以下评测维度：
    1. 幻觉（Hallucination）
    2. 上下文利用率（Context Utilization）
    3. 噪音敏感度（Noise Sensitivity）
    4. 自身知识（Self-knowledge）
    5. 忠实度（Faithfulness）
    """
    return f"""请基于以下{domain}领域的文档内容，生成 {n} 个高质量的 RAG 系统评测问答样本。

【领域文档内容】
{corpus_text[:8000]}

【生成要求】

输出 JSON 数组格式，每个元素包含以下字段：
- "query": 用户问题（中文）
- "context": 该问题对应的理想参考上下文（从文档中摘录的相关段落，100-300字）
- "answer": 基于context的标准答案（中文）
- "dimension": 该样本主要评测的维度，从以下5种中选择：
  - "hallucination": 问题设计为容易诱使RAG系统生成文档中不存在的内容。标准答案必须严格限定在context范围内。
  - "context_utilization": 问题需要综合context中多个分散信息点才能完整回答，测试系统能否有效整合上下文。
  - "noise_sensitivity": 问题的关键词在文档中多处出现（包括相关和不相关段落），测试系统能否排除噪音、定位真正相关信息。
  - "self_knowledge": 问题涉及的概念是通用电力知识（如"什么是短路电流"），文档中也有提及。测试系统是否优先使用检索上下文而非自身知识。
  - "faithfulness": 问题的答案可以直接在context中找到明确依据，测试系统是否忠实引用而非自由发挥。
- "difficulty": 难度等级，"easy" / "medium" / "hard"

【质量要求】
1. 每个问题的答案必须能在提供的文档内容中找到依据，不能编造。
2. 问题要具体、有针对性，避免泛泛而问（如"这篇文章讲了什么"）。
3. 不同样本应覆盖不同文档、不同章节，避免重复。
4. 难度分布：easy 30%, medium 50%, hard 20%。
5. 输出必须是合法的 JSON 数组，不要包含 markdown 代码块标记（```json）。

请直接输出 JSON 数组：
"""


def parse_qa_json(raw_text: str) -> List[Dict[str, Any]]:
    """解析 LLM 返回的 JSON"""
    # 去掉可能的 markdown 代码块标记
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # 去掉第一行 ```json 和最后一行 ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[警告] JSON 解析失败: {e}")
        print(f"原始文本前500字:\n{text[:500]}")
        return []


def generate_domain_qa(domain: str, n: int) -> List[Dict[str, Any]]:
    """为单个领域生成 QA 样本"""
    print(f"\n[{domain}] 加载语料...")
    corpus = load_corpus(domain)

    print(f"[{domain}] 调用 LLM 生成 {n} 个 QA 样本...")
    prompt = generate_qa_prompt(corpus, domain, n)
    raw_response = call_llm(prompt)

    qa_list = parse_qa_json(raw_response)
    print(f"[{domain}] 成功生成 {len(qa_list)} 个样本")

    # 补充来源信息
    for qa in qa_list:
        qa["domain"] = domain

    return qa_list


def main():
    parser = argparse.ArgumentParser(description="生成 RAG 评测问答样本")
    parser.add_argument("--domain", default="all", help="领域名称，如 protection / dispatch / distribution / equipment，或 all")
    parser.add_argument("--n", type=int, default=10, help="每个领域生成的样本数量")
    parser.add_argument("--output", default=None, help="输出文件路径（可选）")
    args = parser.parse_args()

    domains = ["dispatch", "protection", "distribution", "equipment"] if args.domain == "all" else [args.domain]

    all_qa = []
    for domain in domains:
        try:
            qa_list = generate_domain_qa(domain, args.n)
            all_qa.extend(qa_list)

            # 按领域保存
            domain_file = OUTPUT_DIR / f"{domain}_qa.json"
            domain_file.write_text(
                json.dumps(qa_list, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"[OK] 已保存: {domain_file}")
        except Exception as e:
            print(f"[错误] {domain} 生成失败: {e}")

    # 保存汇总
    if len(domains) > 1:
        all_file = OUTPUT_DIR / "all_qa.json"
        all_file.write_text(
            json.dumps(all_qa, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\n[OK] 汇总已保存: {all_file} ({len(all_qa)} 条)")


if __name__ == "__main__":
    main()
