# rag_system.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# ============ 配置区 ============
DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"
MODEL_NAME = "qwen2.5:3b"
# 换成一个轻量级、可以本地运行的模型
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============ 检查文档是否存在 ============
if not os.path.exists(f"{DOCS_DIR}/campus_info.txt"):
    print("❌ 错误：请先在 docs 文件夹下创建 campus_info.txt 文件！")
    exit()

# ============ 第一步：加载文档 ============
print("📂 正在加载文档...")
loader = TextLoader(f"{DOCS_DIR}/campus_info.txt", encoding="utf-8")
documents = loader.load()
print(f"✅ 加载了 {len(documents)} 个文档")

# ============ 第二步：切分文档 ============
print("✂️ 正在切分文档...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"✅ 切分成了 {len(chunks)} 个文本片段")

for i, chunk in enumerate(chunks[:3]):
    print(f"  片段{i+1}: {chunk.page_content[:50]}...")

# ============ 第三步：向量化并存入FAISS ============
print("🔢 正在生成向量并建立索引...")
# 使用本地轻量级模型，不需要联网下载
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(INDEX_DIR)
print(f"✅ 向量索引已保存到 {INDEX_DIR}")

# ============ 第四步：问答函数 ============
def ask_question(question):
    print(f"\n❓ 问题: {question}")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    
    print(f"📄 检索到 {len(docs)} 个相关片段:")
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] {doc.page_content[:80]}...")
    
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""你是一个校园助手，请只根据以下参考资料回答问题。如果参考资料中没有相关信息，请说"资料中没有提到"。

参考资料：
{context}

问题：{question}
回答："""

    llm = Ollama(model=MODEL_NAME)
    response = llm.invoke(prompt)
    
    print(f"💬 回答: {response}")
    return response

# ============ 第五步：运行测试 ============
if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    
    ask_question("图书馆什么时候开门？")
    ask_question("怎么申请奖学金？")
    ask_question("如何补办学生证？")