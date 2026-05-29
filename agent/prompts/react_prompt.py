"""
第7课（Part 3）：ReAct 提示词模板

ReAct（Reasoning + Acting）的核心是构造一个循环提示词，
让 LLM 在"思考"和"行动"之间交替，直到得出结论。

这个模块提供标准化的 ReAct 提示词模板。
"""

from typing import List, Dict, Any


REACT_SYSTEM_PROMPT = """你是一个智能助手，可以通过推理和调用工具来回答用户问题。

可用工具：
{tool_descriptions}

请严格遵循以下格式输出：

思考：分析当前状态，决定下一步行动
行动：工具名称
行动输入：{"参数名": "参数值"}
观察：工具返回的结果（由系统填入）

重复"思考→行动→观察"循环，直到你能直接回答用户问题。

最终回答格式：
思考：我已经有足够的信息回答问题
最终答案：你的回答

重要规则：
1. 如果知识库检索不到相关信息，明确告知用户
2. 涉及数字计算时，优先使用 calculator 工具
3. 不要编造工具返回中没有的信息
"""


class ReActPromptBuilder:
    """
    ReAct 提示词构建器

    动态拼接工具描述、对话历史、中间结果，构造完整的提示词。
    """

    @staticmethod
    def build_system_prompt(tools: List[Dict[str, Any]]) -> str:
        """
        构建系统提示词（包含工具描述）

        Args:
            tools: 工具信息列表，每个工具包含 name、description、input_schema
        """
        tool_descs = []
        for tool in tools:
            desc = f"- {tool['name']}: {tool['description']}\n  参数: {tool['input_schema']}"
            tool_descs.append(desc)

        return REACT_SYSTEM_PROMPT.format(
            tool_descriptions="\n".join(tool_descs)
        )

    @staticmethod
    def build_user_prompt(query: str, history: List[Dict[str, str]] = None) -> str:
        """
        构建用户提示词

        Args:
            query: 用户当前问题
            history: 多轮对话历史 [{"role": "user/assistant", "content": "..."}]
        """
        if history:
            history_text = "\n".join([
                f"{'用户' if h['role'] == 'user' else '助手'}: {h['content']}"
                for h in history
            ])
            return f"对话历史：\n{history_text}\n\n当前问题：{query}"
        return query

    @staticmethod
    def build_observation(tool_name: str, result: Any) -> str:
        """
        构建观察提示词（工具执行结果）

        这个结果会追加到提示词中，让 LLM 基于最新观察继续推理。
        """
        return f"观察：工具 '{tool_name}' 返回结果：\n{result}\n"


# 直接问答提示词（不走 ReAct，用于简单问题）
DIRECT_QA_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。

回答规则：
1. 基于提供的参考资料回答问题
2. 如果参考资料中没有答案，明确说"无法从现有知识库中找到答案"
3. 引用来源时使用 [1], [2] 等编号
4. 保持回答简洁准确
"""

# 意图识别提示词
INTENT_RECOGNITION_PROMPT = """分析用户查询的意图，从以下类别中选择最匹配的一个：

factual_qa - 事实性问答（如"什么是XX""怎么操作YY"）
summarization - 摘要总结（如"总结一下""主要内容是什么"）
comparison - 对比分析（如"A和B有什么区别"）
multi_step - 多步推理（需要多个步骤才能回答）
tool_call - 需要外部工具（如天气查询、计算、数据库查询）
chitchat - 闲聊问候

只输出类别名称，不要解释。查询：{query}"""
