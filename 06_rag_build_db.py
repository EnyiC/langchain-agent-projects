import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 加载
data_dir = "data"
all_docs = []
for file_path in glob.glob(os.path.join(data_dir, "*")):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        continue
    docs = loader.load()
    company = os.path.basename(file_path).replace(".txt", "").replace(".pdf", "")
    for doc in docs:
        doc.metadata["company"] = company
        doc.metadata["source"] = file_path
    all_docs.extend(docs)

# 切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(all_docs)

# 向量化
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 重建（先删除旧的）
import shutil
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"✅ 向量数据库已重建，共存入 {len(chunks)} 个文本块")