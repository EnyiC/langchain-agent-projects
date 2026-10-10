import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ← 修正

# 1. 加载文档
file_path = "data/sample.txt"
if file_path.endswith(".pdf"):
    loader = PyPDFLoader(file_path)
else:
    loader = TextLoader(file_path, encoding="utf-8")

documents = loader.load()
print(f"✅ 加载了 {len(documents)} 页文档")

# 2. 文本切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"✅ 切分成 {len(chunks)} 个文本块")
print(f"第一个块的内容预览：\n{chunks[0].page_content[:200]}")