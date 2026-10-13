# 智能财报分析助手（RAG）

## 项目背景
基于 LangChain + ChromaDB + DeepSeek，构建一个能读取多份财报文档、支持元数据过滤的智能问答系统。

## 技术栈
- LangChain（文档加载、切分、RAG 链）
- ChromaDB（向量数据库）
- HuggingFace Embeddings（本地向量化）
- DeepSeek API（LLM 生成）
- Python

## 核心功能
1. 批量加载 PDF/TXT 文档，自动添加公司元数据
2. 文本切分 + 向量化存储
3. 支持 k=5 检索增强
4. 支持按公司名过滤检索
5. 交互式问答循环

## 运行方法
```bash
python 05_rag_load_and_split.py   # 加载与切分
python 06_rag_build_db.py         # 构建向量库
python 07_rag_qa.py               # 问答