import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.0
)

parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_template(
    "请分析以下新闻的情感，输出 JSON，包含字段：sentiment（利好/利空/中性）、score（-1到1）、reason。新闻：{news}"
)

chain = prompt | llm | parser

result = chain.invoke({"news": "美联储宣布降息50个基点，美股大涨。"})
print(result)