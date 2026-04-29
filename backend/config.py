"""
系统配置
"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
NOVELS_DIR = os.path.join(DATA_DIR, "novels")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

# 阿里云API配置（从环境变量读取，本地开发使用默认值）
import os
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

# 模型配置
EMBEDDING_MODEL = "text-embedding-v2"  # 阿里云Embedding模型
LLM_MODEL = "qwen-turbo-latest"  # 阿里云大模型

# 文本分块配置
CHUNK_SIZE = 500  # 每个文本块的最大字符数
CHUNK_OVERLAP = 100  # 文本块重叠字符数

# 检索配置
TOP_K = 5  # 检索返回的最相似文本块数量

# 确保目录存在
for dir_path in [NOVELS_DIR, METADATA_DIR, CHROMA_DIR]:
    os.makedirs(dir_path, exist_ok=True)
