import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# 1. 定义共享白板（State）
class NewsState(TypedDict):
    news: str
    sentiment_score: float
    summary: str

# 2. 初始化 LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.0
)

# 3. 定义节点（干活的人）
def analyze_sentiment(state: NewsState):
    """节点1：情感打分"""
    news = state["news"]
    prompt = f"请判断以下金融新闻的情感，只输出一个数字（-1到1之间）：\n{news}"
    result = llm.invoke(prompt)
    try:
        score = float(result.content.strip())
    except:
        score = 0.0
    print(f"[情感分析节点] 新闻得分：{score}")
    return {"sentiment_score": score}

def summarize_bullish(state: NewsState):
    """节点2A：利好总结"""
    print("[利好总结节点] 触发")
    prompt = f"请基于以下新闻写一句简短的利好总结：\n{state['news']}"
    result = llm.invoke(prompt)
    return {"summary": f"【利好】{result.content}"}

def summarize_bearish(state: NewsState):
    """节点2B：利空总结"""
    print("[利空总结节点] 触发")
    prompt = f"请基于以下新闻写一句简短的利空总结：\n{state['news']}"
    result = llm.invoke(prompt)
    return {"summary": f"【利空】{result.content}"}

# 4. 定义条件边（十字路口）
def route_sentiment(state: NewsState) -> Literal["bullish", "bearish"]:
    if state["sentiment_score"] > 0:
        return "bullish"
    else:
        return "bearish"

# 5. 构建图
builder = StateGraph(NewsState)

# 添加节点
builder.add_node("analyze", analyze_sentiment)
builder.add_node("bullish", summarize_bullish)
builder.add_node("bearish", summarize_bearish)

# 添加边（连线）
builder.add_edge(START, "analyze")
builder.add_conditional_edges("analyze", route_sentiment, {"bullish": "bullish", "bearish": "bearish"})
builder.add_edge("bullish", END)
builder.add_edge("bearish", END)

# 编译图
graph = builder.compile()

# 6. 运行测试
print("=== 测试1：利好新闻 ===")
result1 = graph.invoke({"news": "美联储宣布降息50个基点，美股大涨。"})
print(f"最终结果：{result1['summary']}\n")

print("=== 测试2：利空新闻 ===")
result2 = graph.invoke({"news": "美国国债收益率升至6%，科技股集体暴跌。"})
print(f"最终结果：{result2['summary']}\n")