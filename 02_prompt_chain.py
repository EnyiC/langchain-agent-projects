import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_template(
    "请帮我写一封求职邮件，应聘 {company} 的 {role} 岗位。我的背景是：{background}。"
)

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "company": "SHEIN",
    "role": "数据分析师",
    "background": "悉尼大学商业硕士，熟练使用 Python、SQL、Power BI，有金融和电商数据分析项目经验。"
})
print(result)