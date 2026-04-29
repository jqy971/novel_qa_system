"""
RAG引擎 - 检索增强生成核心逻辑
整合向量检索与大模型生成，支持FastGPT风格的提示词工程
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from backend.services.vector_store import VectorStore
from backend.services.llm_client import QwenClient
from backend.prompts.rag_prompts import (
    SYSTEM_PROMPT, 
    build_rag_prompt,
    get_prompt_template
)
from backend.config import TOP_K


@dataclass
class RAGResult:
    """RAG结果"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    intent: str


class RAGEngine:
    """RAG引擎"""
    
    def __init__(self, novel_id: str):
        self.novel_id = novel_id
        self.vector_store = VectorStore(novel_id)
        self.llm_client = QwenClient()
    
    def answer(
        self,
        query: str,
        intent: str = "qa",
        top_k: int = TOP_K,
        temperature: float = 0.7,
        history: Optional[List[Dict]] = None
    ) -> RAGResult:
        """
        基于RAG回答问题
        
        Args:
            query: 用户问题
            intent: 意图类型
            top_k: 检索数量
            temperature: 生成温度
            history: 对话历史
            
        Returns:
            RAG结果
        """
        # 1. 检索相关文本块
        print(f"[RAG] 正在检索与问题相关的文本: {query}")
        
        stats = self.vector_store.get_stats()
        print(f"[RAG] 向量库状态: {stats}")
        
        retrieved_chunks = self.vector_store.search(query, top_k=top_k)
        print(f"[RAG] 检索到 {len(retrieved_chunks)} 个相关文本块")
        
        # 2. 检查向量库状态
        if not retrieved_chunks:
            if stats.get('total_chunks', 0) == 0:
                return RAGResult(
                    answer="📚 **向量库为空**\n\n请先上传并处理小说文件，我才能为你提供问答服务。",
                    sources=[],
                    query=query,
                    intent=intent
                )
            return RAGResult(
                answer="🔍 **未找到相关内容**\n\n抱歉，在小说中未找到与问题相关的内容。请尝试换一种问法。",
                sources=[],
                query=query,
                intent=intent
            )
        
        # 3. 构建上下文
        context = self._build_context(retrieved_chunks)
        print(f"[RAG] 构建上下文完成，长度: {len(context)} 字符")
        
        # 4. 构建提示词
        prompt = build_rag_prompt(
            query=query,
            context=context,
            intent=intent,
            history=history
        )
        
        # 5. 生成回答
        print("[RAG] 正在生成回答...")
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.chat(messages, temperature=temperature)
            answer = response.content
            
        except Exception as e:
            print(f"[RAG] LLM调用失败: {e}")
            # 使用本地回答（基于检索结果）
            answer = self._local_answer(query, retrieved_chunks)
        
        # 6. 格式化来源
        sources = self._format_sources(retrieved_chunks)
        
        return RAGResult(
            answer=answer,
            sources=sources,
            query=query,
            intent=intent
        )
    
    def answer_stream(
        self,
        query: str,
        intent: str = "qa",
        top_k: int = TOP_K,
        history: Optional[List[Dict]] = None
    ):
        """
        流式回答
        
        Args:
            query: 用户问题
            intent: 意图类型
            top_k: 检索数量
            history: 对话历史
            
        Yields:
            生成的文本片段
        """
        # 1. 检索
        retrieved_chunks = self.vector_store.search(query, top_k=top_k)
        
        if not retrieved_chunks:
            stats = self.vector_store.get_stats()
            if stats.get('total_chunks', 0) == 0:
                yield "📚 **向量库为空**\n\n请先上传并处理小说文件。"
            else:
                yield "🔍 **未找到相关内容**\n\n抱歉，在小说中未找到与问题相关的内容。"
            return
        
        # 2. 构建上下文和提示词
        context = self._build_context(retrieved_chunks)
        prompt = build_rag_prompt(
            query=query,
            context=context,
            intent=intent,
            history=history
        )
        
        # 3. 流式生成
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        try:
            for chunk in self.llm_client.chat_stream(messages):
                yield chunk
        except Exception as e:
            print(f"[RAG] 流式生成失败: {e}")
            yield self._local_answer(query, retrieved_chunks)
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """构建上下文文本"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            chapter = chunk["metadata"].get("chapter_title", "未知章节")
            content = chunk["content"]
            similarity = chunk.get("similarity", 0)
            context_parts.append(f"[片段{i}] {chapter} (相关度: {similarity:.2%})\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化来源信息"""
        sources = []
        for chunk in chunks:
            sources.append({
                "chapter": chunk["metadata"].get("chapter_title", "未知章节"),
                "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                "similarity": chunk.get("similarity", 0),
                "chunk_id": chunk["id"]
            })
        return sources
    
    def _local_answer(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        本地回答（当LLM不可用时使用）
        直接返回最相关的文本片段
        """
        if not chunks:
            return "抱歉，未找到相关内容。"
        
        # 取最相关的片段
        best_chunk = chunks[0]
        chapter = best_chunk["metadata"].get("chapter_title", "未知章节")
        content = best_chunk["content"]
        
        return f"根据小说原文：\n\n【{chapter}】\n{content[:500]}...\n\n（以上为原文内容）"
    
    def get_relevant_context(
        self,
        query: str,
        top_k: int = TOP_K
    ) -> List[Dict[str, Any]]:
        """仅获取相关上下文"""
        return self.vector_store.search(query, top_k=top_k)


class ContinueEngine:
    """续写引擎"""
    
    def __init__(self, novel_id: str):
        self.novel_id = novel_id
        self.vector_store = VectorStore(novel_id)
        self.llm_client = QwenClient()
    
    def continue_story(
        self,
        query: str,
        context_length: int = 1000,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        续写小说
        
        Args:
            query: 用户指令
            context_length: 检索的上下文长度
            max_tokens: 最大生成长度
            
        Returns:
            包含续写内容和参考来源的字典
        """
        # 1. 检索相关上下文
        retrieved_chunks = self.vector_store.search(query, top_k=5)
        
        if not retrieved_chunks:
            return {
                "continuation": "📚 **无法续写**\n\n未找到相关上下文，请先上传小说文件。",
                "sources": []
            }
        
        # 2. 构建上下文和风格样本
        context_text = ""
        style_sample = ""
        
        for chunk in retrieved_chunks:
            content = chunk["content"]
            if len(style_sample) < 500:
                style_sample += content + "\n"
            context_text += content + "\n"
            if len(context_text) >= context_length:
                break
        
        # 3. 构建提示词
        from backend.prompts.rag_prompts import CONTINUE_WRITING_PROMPT
        
        prompt = CONTINUE_WRITING_PROMPT.format(
            style_sample=style_sample[:500],
            context=context_text[:context_length],
            instruction=query
        )
        
        # 4. 生成续写
        print("[RAG] 正在生成续写内容...")
        try:
            messages = [
                {"role": "system", "content": "你是一个专业的小说续写助手。"},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.chat(messages, temperature=0.8, max_tokens=max_tokens)
            continuation = response.content
            
        except Exception as e:
            print(f"[RAG] 续写生成失败: {e}")
            continuation = "⚠️ **续写服务暂时不可用**\n\n请检查API配置或稍后重试。"
        
        return {
            "continuation": continuation,
            "sources": [
                {
                    "chapter": chunk["metadata"].get("chapter_title", "未知章节"),
                    "content_preview": chunk["content"][:150] + "..."
                }
                for chunk in retrieved_chunks[:3]
            ]
        }
