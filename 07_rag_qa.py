import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 加载向量库
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 检索器：k=5，提高召回
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.0
)

prompt = ChatPromptTemplate.from_template(
    """基于以下上下文回答问题。如果上下文中没有相关信息，请说“文档中未提及”。
上下文：
{context}

问题：{question}
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 交互式问答循环
print("智能财报分析助手已启动（输入 quit 退出）")
while True:
    question = input("\n你的问题：")
    if question.lower() == "quit":
        break
    answer = rag_chain.invoke(question)
    print(f"回答：{answer}")

# 在 rag_chain 定义之前加入
def get_company_retriever(company):
    return vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": {"company": company}}
    )

# 交互式循环改为
print("智能财报分析助手已启动（输入 quit 退出）")
while True:
    company = input("\n请输入公司名（如 sample1/sample2，或直接回车不筛选）：").strip()
    question = input("你的问题：")
    if question.lower() == "quit":
        break
    if company:
        r = get_company_retriever(company)
        chain = (
            {"context": r | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
    else:
        chain = rag_chain
    answer = chain.invoke(question)
    print(f"回答：{answer}")