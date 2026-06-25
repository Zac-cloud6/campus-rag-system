# rag_gradio.py
# RAG 校园问答系统 - Gradio Web 界面版（支持多文档）

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import gradio as gr

# ============ 配置 ============
DOCS_DIR = "docs"
INDEX_DIR = "faiss_index"
MODEL_NAME = "qwen2.5:3b"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ============ 加载所有文档（多文档支持） ============
def load_all_documents():
    """加载 docs 文件夹下所有 .txt 文件"""
    all_documents = []
    txt_files = glob.glob(f"{DOCS_DIR}/*.txt")
    
    if not txt_files:
        print(f"❌ 错误：{DOCS_DIR} 文件夹下没有找到 .txt 文件")
        return []
    
    print(f"📂 找到 {len(txt_files)} 个文档文件")
    for file_path in txt_files:
        print(f"  正在加载：{os.path.basename(file_path)}")
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        all_documents.extend(documents)
    
    print(f"✅ 共加载 {len(all_documents)} 个文档")
    return all_documents

# ============ 构建或加载向量索引 ============
def build_or_load_index():
    """如果索引已存在则加载，否则重新构建"""
    embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        'device': 'cpu',
        'local_files_only': True  # <--- 添加这一行，强制使用本地文件
    },
    encode_kwargs={'normalize_embeddings': True}
)
    
    if os.path.exists(INDEX_DIR):
        print(f"📂 加载已有索引：{INDEX_DIR}")
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    else:
        print("🔨 索引不存在，重新构建...")
        documents = load_all_documents()
        if not documents:
            return None
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✂️ 切分成 {len(chunks)} 个文本片段")
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_DIR)
        print(f"✅ 索引已保存到 {INDEX_DIR}")
        return vectorstore

# ============ 问答函数（供 Gradio 调用） ============
def ask_question(question, history=None):
    """输入问题，返回回答和检索到的文档片段"""
    if not question or question.strip() == "":
        return "请输入一个有效的问题", ""
    
    try:
        # 加载索引
        vectorstore = build_or_load_index()
        if vectorstore is None:
            return "❌ 没有找到文档，请在 docs 文件夹下放入 .txt 文件", ""
        
        # 检索相关文档
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)
        
        # 构建上下文
        context = "\n".join([doc.page_content for doc in docs])
        
        # 构建提示词
        prompt = f"""你是一个校园助手，请只根据以下参考资料回答问题。如果参考资料中没有相关信息，请说"资料中没有提到"。

参考资料：
{context}

问题：{question}
回答："""

        # 调用大模型
        llm = Ollama(model=MODEL_NAME)
        response = llm.invoke(prompt)
        
        # 构建检索到的片段展示
        retrieved_info = f"📄 检索到 {len(docs)} 个相关片段：\n\n"
        for i, doc in enumerate(docs):
            retrieved_info += f"【片段 {i+1}】\n{doc.page_content}\n\n"
        
        return response, retrieved_info
        
    except Exception as e:
        return f"❌ 出错：{str(e)}", ""

# ============ 创建 Gradio 界面 ============
def create_interface():
    with gr.Blocks(title="校园 RAG 问答系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏫 校园生活助手 RAG 智能问答系统
        
        基于 **LangChain + FAISS + Ollama** 实现，完全本地运行。
        
        💡 **提示**：在 `docs/` 文件夹下放入你的校园文档（.txt 格式），系统会自动加载。
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="💬 输入你的问题",
                    placeholder="例如：图书馆什么时候开门？",
                    lines=2
                )
                submit_btn = gr.Button("🚀 提问", variant="primary")
                clear_btn = gr.Button("🗑️ 清空")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📚 支持的文档")
                file_list = gr.File(
                    file_count="multiple",
                    label="查看 docs 文件夹中的文档",
                    interactive=False
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                answer_output = gr.Textbox(
                    label="💬 回答",
                    lines=8,
                    interactive=False
                )
            with gr.Column(scale=1):
                retrieved_output = gr.Textbox(
                    label="📄 检索到的文档片段",
                    lines=8,
                    interactive=False
                )
        
        # 绑定事件
        submit_btn.click(
            fn=ask_question,
            inputs=[question_input],
            outputs=[answer_output, retrieved_output]
        )
        
        question_input.submit(
            fn=ask_question,
            inputs=[question_input],
            outputs=[answer_output, retrieved_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            outputs=[question_input, answer_output, retrieved_output]
        )
        
        # 加载示例
        gr.Markdown("""
        ### 📌 试试这些问题
        - 图书馆什么时候开门？
        - 怎么申请奖学金？
        - 如何补办学生证？
        - 校园卡怎么充值？
        """)
    
    return demo

# ============ 主程序 ============
if __name__ == "__main__":
    print("🤖 启动 RAG 校园问答系统 (Gradio 版)")
    print("=" * 50)
    
    # 检查 docs 文件夹
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"📁 已创建 {DOCS_DIR} 文件夹，请放入 .txt 文档")
    
    # 检查 Ollama
    try:
        llm = Ollama(model=MODEL_NAME)
        llm.invoke("测试")
        print("✅ Ollama 连接正常")
    except:
        print("❌ 请先启动 Ollama 服务（任务栏羊驼图标）")
        print("   Ollama 启动后，重新运行本程序")
        input("按 Enter 键退出...")
        exit()
    
    # 预构建索引
    print("📂 正在加载/构建索引...")
    build_or_load_index()
    print("✅ 准备就绪！")
    
    # 启动 Gradio
    print("\n🚀 启动 Web 界面...")
    print("   在浏览器中打开：http://127.0.0.1:7860")
    print("   按 Ctrl+C 停止服务\n")
    
    demo = create_interface()
    demo.launch(share=False)