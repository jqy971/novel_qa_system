# 📚 智能辅助阅读系统

基于RAG（检索增强生成）的单本网络小说问答与辅助阅读系统。

## 系统特点

- **严格基于原文**：所有回答都基于小说原文检索，避免大模型幻觉
- **智能意图识别**：自动识别用户是想问答、续写还是摘要
- **人物关系分析**：自动抽取人物并分析关系
- **情节续写**：基于原著风格智能续写

## 技术架构

```
前端 (HTML + JavaScript)
    ↓
后端 (FastAPI + Python 3.10)
    ↓
Agent调度 → Skills执行
    ↓
向量库 (ChromaDB) + 阿里云API (qwen-turbo)
```

## 涉及的NLP任务

1. **文本分割与结构化** - 章节识别、语义分块
2. **文本向量表示与语义检索** - Embedding、相似度匹配
3. **意图识别** - 区分问答/续写/摘要等意图
4. **信息抽取** - 人物识别、关系抽取
5. **文本生成** - 问答生成、情节续写

## 快速开始

### 1. 配置API Key

编辑 `backend/config.py`，设置你的阿里云API Key：

```python
DASHSCOPE_API_KEY = "your-api-key-here"  # 替换为你的Key
```

获取地址：https://dashscope.aliyun.com/

### 2. 启动系统

```bash
python start.py
```

或手动启动：

```bash
# 安装依赖
pip install -r requirements.txt

# 启动后端
cd backend
python -m uvicorn app:app --reload

# 前端直接打开 frontend/index.html
```

### 3. 使用

1. 在浏览器中打开前端页面
2. 上传TXT格式的小说文件
3. 等待处理完成（自动分块、生成向量）
4. 开始提问或要求续写

## 使用示例

**问答**：
- "萧炎和美杜莎后来怎么样了？"
- "主角什么时候突破斗帝的？"

**续写**：
- "帮我续写萧炎遇到美杜莎之后的故事"
- "如果萧炎没有退婚，后面会怎样？"

**摘要**：
- "总结一下第一章"
- "这篇小说主要讲了什么"

## 项目结构

```
novel_qa_system/
├── backend/
│   ├── app.py              # FastAPI主应用
│   ├── config.py           # 配置文件
│   ├── agent.py            # Agent调度
│   ├── services/
│   │   ├── novel_parser.py # 小说解析
│   │   ├── vector_store.py # 向量存储
│   │   ├── llm_client.py   # LLM客户端
│   │   └── rag_engine.py   # RAG引擎
│   └── skills/
│       ├── intent_classifier.py  # 意图识别
│       ├── qa_skill.py           # 问答
│       ├── continue_skill.py     # 续写
│       └── extract_skill.py      # 信息抽取
├── frontend/
│   └── index.html          # 前端页面
├── data/                   # 数据存储
├── requirements.txt
├── start.py
└── README.md
```

## 注意事项

1. 首次上传小说需要等待向量生成，时间取决于小说长度
2. 确保网络连接正常（需要调用阿里云API）
3. 建议使用UTF-8编码的TXT文件

## 许可证

MIT License
