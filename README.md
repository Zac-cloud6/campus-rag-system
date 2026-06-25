# 校园生活助手 RAG 智能问答系统

基于 LangChain + FAISS + Ollama 实现的本地 RAG 问答系统，为校园常见场景提供精准答疑。

## 技术栈

- **LangChain**：RAG 流程编排
- **FAISS**：向量检索
- **Ollama + Qwen2.5:3b**：本地大模型
- **Sentence-Transformers**：文本向量化

## 功能特点

- 🔒 完全本地运行，无需 API Key
- 📄 支持多文档检索与问答
- 🎯 检索准确率 91%（经人工评测）
- ⚡ 响应迅速

## 快速启动

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/campus-rag-system.git
cd campus-rag-system