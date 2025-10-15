#!/bin/bash
echo "🚀 正在初始化 RAG 环境..."

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 检查 Ollama 是否安装
if ! command -v ollama &> /dev/null
then
    echo "❌ 未检测到 Ollama，请先安装：https://ollama.com/download"
    exit
fi

# 下载 Mistral 模型
ollama pull mistral

# 构建向量索引
python build_index.py

# 启动 Streamlit 应用
streamlit run app.py
