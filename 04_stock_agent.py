import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool          # ← 从 langchain_core 导入
from langchain.agents import create_agent      # ← 新版 agent 入口
import akshare as ak

load_dotenv()

@tool
def get_stock_price(symbol: str) -> str:
    """输入股票代码，返回最新股价。例如：000001"""
    # 模拟数据，用于测试 Agent 工具调用逻辑（真实场景再换成 API 调用）
    mock_data = {
        "000001": 11.52,
        "600519": 1680.50,
        "300750": 185.30
    }
    price = mock_data.get(symbol, "未知")
    if price != "未知":
        return f"{symbol} 最新价：{price}"
    return f"未找到 {symbol} 的股价数据"

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.0
)

# 创建 agent（新版 API 会自动处理工具调用循环）
agent = create_agent(
    model=llm,
    tools=[get_stock_price],
    system_prompt="你是一个股票助手。当用户询问股价时，使用 get_stock_price 工具查询。"
)

# 调用 agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我查一下平安银行（000001）现在的股价"}]}
)

# 打印最终回复
print(result["messages"][-1].content)