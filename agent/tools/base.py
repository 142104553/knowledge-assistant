"""
第7课（Part 2）：工具基类与示例工具

Agent 的强大之处在于可以调用外部工具扩展能力。
每个工具是一个独立的可调用单元，有明确的输入输出规范。

核心接口：
    tool = CalculatorTool()
    result = tool.execute(expression="2 + 2")
    # 返回: ToolResult(output=4, success=True)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str
    input_params: Dict[str, Any]
    output: Any
    success: bool
    error_message: str = ""


class BaseTool(ABC):
    """
    工具抽象基类

    每个工具必须定义：
    - name: 工具名称（Agent 通过名称调用）
    - description: 功能描述（LLM 通过描述理解何时使用该工具）
    - input_schema: 输入参数规范（JSON Schema 格式）
    - execute: 执行逻辑
    """

    name: str = ""
    description: str = ""
    input_schema: Dict = {}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass

    def get_tool_info(self) -> Dict[str, Any]:
        """返回工具信息，用于 LLM 的工具选择决策"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }


class CalculatorTool(BaseTool):
    """
    计算器工具

    用于执行数学运算。LLM 在处理涉及数字的问题时，
    直接推理容易出错（如大数乘法），调用计算器可以保证准确性。
    """

    name = "calculator"
    description = "执行数学计算，如加减乘除、百分比等。输入是数学表达式字符串。"
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '123 * 456' 或 '(100 - 20) / 5'"
            }
        },
        "required": ["expression"]
    }

    def execute(self, expression: str) -> ToolResult:
        try:
            # 安全评估：只允许数字和基本运算符
            allowed_chars = set("0123456789+-*/.() %")
            if not all(c in allowed_chars for c in expression.replace(" ", "")):
                raise ValueError("表达式包含非法字符")

            result = eval(expression)  # 在生产环境中应使用更安全的解析器，如 numexpr
            return ToolResult(
                tool_name=self.name,
                input_params={"expression": expression},
                output=result,
                success=True
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                input_params={"expression": expression},
                output=None,
                success=False,
                error_message=str(e)
            )


class DatabaseQueryTool(BaseTool):
    """
    数据库查询工具（示例）

    用于执行结构化数据查询。实际项目中可连接 MySQL/PostgreSQL/ClickHouse 等。
    这里用模拟数据演示。
    """

    name = "database_query"
    description = "查询结构化数据库，如销售数据、用户数据等。输入是 SQL 查询语句。"
    input_schema = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "SQL 查询语句"
            }
        },
        "required": ["sql"]
    }

    def __init__(self):
        # 模拟数据库：产品销售额数据
        self.mock_data = {
            "sales": [
                {"product": "产品A", "month": "2024-01", "amount": 150000},
                {"product": "产品B", "month": "2024-01", "amount": 230000},
                {"product": "产品C", "month": "2024-01", "amount": 89000},
                {"product": "产品A", "month": "2024-02", "amount": 180000},
                {"product": "产品B", "month": "2024-02", "amount": 210000},
            ]
        }

    def execute(self, sql: str) -> ToolResult:
        # 实际项目中这里会连接真实数据库执行 SQL
        # 简化演示：解析简单的 SELECT 语句
        try:
            sql_lower = sql.lower()
            if "select" in sql_lower and "from sales" in sql_lower:
                # 模拟查询结果
                results = self.mock_data["sales"]
                return ToolResult(
                    tool_name=self.name,
                    input_params={"sql": sql},
                    output=results,
                    success=True
                )
            else:
                return ToolResult(
                    tool_name=self.name,
                    input_params={"sql": sql},
                    output=[],
                    success=True
                )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                input_params={"sql": sql},
                output=None,
                success=False,
                error_message=str(e)
            )


class ToolRegistry:
    """
    工具注册表

    管理所有可用工具，Agent 通过名称查找和调用。
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """注册工具"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """获取工具"""
        if name not in self._tools:
            raise ValueError(f"未知工具: {name}")
        return self._tools[name]

    def list_tools(self) -> list:
        """列出所有工具信息"""
        return [tool.get_tool_info() for tool in self._tools.values()]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """执行指定工具"""
        tool = self.get(tool_name)
        return tool.execute(**kwargs)


# 预置工具集
DEFAULT_TOOLS = ToolRegistry()
DEFAULT_TOOLS.register(CalculatorTool())
DEFAULT_TOOLS.register(DatabaseQueryTool())
