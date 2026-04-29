"""
FastAPI 后端主应用
"""
import os
import sys
import uuid
import json
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import NOVELS_DIR, METADATA_DIR
from backend.services.novel_parser import NovelParser
from backend.services.vector_store import VectorStore, VectorStoreManager
from backend.agent import AgentManager, NovelAgent

# 创建FastAPI应用
app = FastAPI(
    title="小说智能问答系统",
    description="基于RAG的单本网络小说问答与辅助阅读系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件（前端）
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
def serve_frontend():
    """提供前端页面"""
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "智能辅助阅读系统 API", "version": "1.0.0", "docs": "/docs"}

# ============ 数据模型 ============

class ChatRequest(BaseModel):
    novel_id: str
    query: str
    stream: bool = False

class ChatResponse(BaseModel):
    success: bool
    intent: str
    answer: str
    sources: Optional[List[dict]] = None

class NovelInfo(BaseModel):
    id: str
    title: str
    chapter_count: int
    chunk_count: int
    total_chars: int
    created_at: str


# ============ API路由 ============

# 删除重复的根路径路由，统一由 serve_frontend 处理


@app.post("/api/novels/upload")
async def upload_novel(file: UploadFile = File(...)):
    """
    上传小说文件
    
    流程：
    1. 保存上传的TXT文件
    2. 解析小说结构（章节分割、文本分块）
    3. 生成向量并存储到向量库
    4. 保存元数据
    """
    # 检查文件类型
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="只支持TXT文件")
    
    # 生成小说ID
    novel_id = str(uuid.uuid4())[:8]
    
    # 保存文件
    file_path = os.path.join(NOVELS_DIR, f"{novel_id}.txt")
    try:
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")
    
    # 解析小说
    try:
        parser = NovelParser(chunk_size=2000, chunk_overlap=200)
        novel_data = parser.parse_file(file_path)
        novel_data["id"] = novel_id
        novel_data["filename"] = file.filename
        novel_data["created_at"] = datetime.now().isoformat()
        
        # 保存元数据
        parser.save_metadata(novel_data, novel_id)
        
        # 构建向量库
        print(f"正在为小说构建向量库: {novel_data['title']}")
        vector_store = VectorStore(novel_id)
        chunks = novel_data.get("chunks", [])
        if chunks:
            success = vector_store.add_chunks(chunks)
            if not success:
                raise Exception("向量库构建失败")
        
        return {
            "success": True,
            "novel_id": novel_id,
            "title": novel_data["title"],
            "chapter_count": novel_data["chapter_count"],
            "chunk_count": novel_data["chunk_count"],
            "message": "小说上传并处理成功"
        }
        
    except Exception as e:
        # 清理文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.get("/api/novels")
def list_novels():
    """获取小说列表"""
    novels = []
    
    for filename in os.listdir(METADATA_DIR):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(METADATA_DIR, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    novels.append({
                        "id": data.get("id"),
                        "title": data.get("title"),
                        "chapter_count": data.get("chapter_count", 0),
                        "chunk_count": data.get("chunk_count", 0),
                        "total_chars": data.get("total_chars", 0),
                        "created_at": data.get("created_at", "")
                    })
            except Exception as e:
                print(f"读取元数据失败 {filename}: {e}")
    
    # 按创建时间排序
    novels.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"novels": novels}


@app.get("/api/novels/{novel_id}")
def get_novel(novel_id: str):
    """获取小说详情"""
    metadata_path = os.path.join(METADATA_DIR, f"{novel_id}.json")
    
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="小说不存在")
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取向量库统计
        vector_store = VectorStore(novel_id)
        stats = vector_store.get_stats()
        
        return {
            "id": data.get("id"),
            "title": data.get("title"),
            "filename": data.get("filename"),
            "chapter_count": data.get("chapter_count", 0),
            "chunk_count": data.get("chunk_count", 0),
            "total_chars": data.get("total_chars", 0),
            "chapters": data.get("chapters", []),
            "vector_stats": stats,
            "created_at": data.get("created_at", "")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取失败: {str(e)}")


@app.delete("/api/novels/{novel_id}")
def delete_novel(novel_id: str):
    """删除小说"""
    errors = []
    
    # 删除文件
    file_path = os.path.join(NOVELS_DIR, f"{novel_id}.txt")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            errors.append(f"删除文件失败: {e}")
    
    # 删除元数据
    metadata_path = os.path.join(METADATA_DIR, f"{novel_id}.json")
    if os.path.exists(metadata_path):
        try:
            os.remove(metadata_path)
        except Exception as e:
            errors.append(f"删除元数据失败: {e}")
    
    # 删除向量库
    try:
        manager = VectorStoreManager()
        manager.delete_collection(novel_id)
    except Exception as e:
        errors.append(f"删除向量库失败: {e}")
    
    # 移除Agent实例
    AgentManager.remove_agent(novel_id)
    
    if errors:
        return {"success": False, "errors": errors}
    
    return {"success": True, "message": "删除成功"}


@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    对话接口 - Agent调度核心
    
    流程：
    1. 获取或创建Agent实例
    2. Agent识别意图并调度Skill
    3. 返回处理结果
    """
    # 检查小说是否存在
    metadata_path = os.path.join(METADATA_DIR, f"{request.novel_id}.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="小说不存在")
    
    # 获取Agent
    agent = AgentManager.get_agent(request.novel_id)
    
    # 处理请求
    result = agent.process(request.query)
    
    if not result["success"]:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": result.get("error", "处理失败"),
                "intent": result.get("intent", "unknown")
            }
        )
    
    response_data = result["result"]
    
    return {
        "success": True,
        "intent": result["intent"],
        "answer": response_data.get("answer") or response_data.get("continuation") or response_data.get("summary", ""),
        "sources": response_data.get("sources", []),
        "type": response_data.get("type", "qa")
    }


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    """流式对话接口"""
    from backend.services.rag_engine import RAGEngine
    
    # 检查小说是否存在
    metadata_path = os.path.join(METADATA_DIR, f"{request.novel_id}.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="小说不存在")
    
    # 使用RAG引擎流式生成
    rag_engine = RAGEngine(request.novel_id)
    
    def generate():
        for chunk in rag_engine.answer_with_stream(request.query):
            yield f"data: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.post("/api/novels/{novel_id}/extract-characters")
def extract_characters(novel_id: str):
    """抽取人物信息"""
    agent = AgentManager.get_agent(novel_id)
    result = agent.extract_characters()
    return result


@app.post("/api/novels/{novel_id}/analyze-relationships")
def analyze_relationships(novel_id: str, characters: List[str]):
    """分析人物关系"""
    agent = AgentManager.get_agent(novel_id)
    result = agent.analyze_relationships(characters)
    return result


@app.get("/api/novels/{novel_id}/chapters")
def get_chapters(novel_id: str):
    """获取章节列表"""
    metadata_path = os.path.join(METADATA_DIR, f"{novel_id}.json")
    
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="小说不存在")
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "chapters": data.get("chapters", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取失败: {str(e)}")


# ============ 启动 ============

if __name__ == "__main__":
    import uvicorn
    print("启动小说智能问答系统...")
    print("API文档: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
