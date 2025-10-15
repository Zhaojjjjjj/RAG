#### 创建虚拟环境

python3 -m venv venv
source venv/bin/activate



#### 安装依赖

pip install -r requirements.txt



#### 下载 Mistral 模型

ollama pull mistral



#### 构建向量索引

python build_index.py



#### 启动 Streamlit 应用

streamlit run app.py
