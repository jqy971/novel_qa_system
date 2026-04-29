"""
向量存储服务 - NLP任务2: 文本向量表示与语义检索
使用本地Embedding模型，无需API费用
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Any

from backend.config import CHROMA_DIR
from backend.services.local_embedding import SimpleEmbedding


class VectorStore:
    """向量数据库管理 - 基于本地Embedding"""
    
    def __init__(self, novel_id: str):
        self.novel_id = novel_id
        self.store_dir = os.path.join(CHROMA_DIR, novel_id)
        self.data_file = os.path.join(self.store_dir, "vectors.pkl")
        
        # 确保目录存在
        os.makedirs(self.store_dir, exist_ok=True)
        
        # 初始化数据
        self.vectors = []
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        # 加载已有数据
        self._load()
        
        # 初始化Embedding模型
        self.embedding_model = SimpleEmbedding(dim=256)
    
    def _load(self):
        """从磁盘加载数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.vectors = data.get('vectors', [])
                    self.documents = data.get('documents', [])
                    self.metadatas = data.get('metadatas', [])
                    self.ids = data.get('ids', [])
                print(f"已加载向量库: {self.novel_id}, 共 {len(self.ids)} 条记录")
            except Exception as e:
                print(f"加载向量库失败: {e}")
    
    def _save(self):
        """保存数据到磁盘"""
        try:
            data = {
                'vectors': self.vectors,
                'documents': self.documents,
                'metadatas': self.metadatas,
                'ids': self.ids
            }
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"保存向量库失败: {e}")
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """添加文本块到向量库"""
        if not chunks:
            return True
        
        total = len(chunks)
        print(f"正在生成 {total} 个文本块的向量...")
        
        # 分批处理
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_chunks = chunks[batch_start:batch_end]
            
            # 提取文本内容
            texts = [chunk["content"] for chunk in batch_chunks]
            
            # 生成本地向量
            print(f"  处理第 {batch_start+1}-{batch_end} 个文本块...")
            embeddings = self.embedding_model.embed_batch(texts)
            
            # 添加到存储
            for i, chunk in enumerate(batch_chunks):
                self.ids.append(chunk["id"])
                self.vectors.append(embeddings[i])
                self.documents.append(chunk["content"])
                self.metadatas.append({
                    "chapter_title": chunk.get("chapter_title", ""),
                    "chapter_index": chunk.get("chapter_index", 0),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "char_count": chunk.get("char_count", 0)
                })
            
            # 每批保存一次
            self._save()
            print(f"  已完成 {batch_end}/{total} 个文本块")
        
        print(f"成功添加 {total} 个文本块到向量库")
        return True
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """语义检索"""
        if not self.vectors:
            return []
        
        # 生成查询向量
        query_vector = self.embedding_model.embed(query)
        
        # 计算相似度
        similarities = []
        for i, vec in enumerate(self.vectors):
            sim = self._cosine_similarity(query_vector, vec)
            similarities.append((i, sim))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k结果
        results = []
        for idx, sim in similarities[:top_k]:
            results.append({
                "id": self.ids[idx],
                "content": self.documents[idx],
                "metadata": self.metadatas[idx],
                "similarity": round(sim, 4)
            })
        
        return results
    
    def delete_collection(self):
        """删除当前小说的向量集合"""
        try:
            import shutil
            if os.path.exists(self.store_dir):
                shutil.rmtree(self.store_dir)
            self.vectors = []
            self.documents = []
            self.metadatas = []
            self.ids = []
            return True
        except Exception as e:
            print(f"删除集合失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计信息"""
        return {
            "novel_id": self.novel_id,
            "total_chunks": len(self.ids)
        }


class VectorStoreManager:
    """向量库管理器"""
    
    def __init__(self):
        os.makedirs(CHROMA_DIR, exist_ok=True)
    
    def list_collections(self) -> List[str]:
        """列出所有小说向量库"""
        if not os.path.exists(CHROMA_DIR):
            return []
        
        collections = []
        for item in os.listdir(CHROMA_DIR):
            item_path = os.path.join(CHROMA_DIR, item)
            if os.path.isdir(item_path):
                if os.path.exists(os.path.join(item_path, "vectors.pkl")):
                    collections.append(item)
        return collections
    
    def delete_collection(self, novel_id: str) -> bool:
        """删除指定小说的向量库"""
        store_dir = os.path.join(CHROMA_DIR, novel_id)
        try:
            import shutil
            if os.path.exists(store_dir):
                shutil.rmtree(store_dir)
            return True
        except Exception as e:
            print(f"删除集合失败: {e}")
            return False
    
    def get_store(self, novel_id: str) -> VectorStore:
        """获取指定小说的向量库实例"""
        return VectorStore(novel_id)
