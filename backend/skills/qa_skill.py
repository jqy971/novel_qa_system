"""
问答Skill - 封装问答能力
"""
from typing import Dict, Any
from backend.services.rag_engine import RAGEngine


class QASkill:
    """问答技能"""
    
    def __init__(self, novel_id: str):
        self.novel_id = novel_id
        self.rag_engine = RAGEngine(novel_id)
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行问答
        
        Args:
            query: 用户问题
            **kwargs: 额外参数
            
        Returns:
            问答结果
        """
        top_k = kwargs.get("top_k", 5)
        temperature = kwargs.get("temperature", 0.7)
        
        # 使用RAG引擎生成回答
        result = self.rag_engine.answer(query, top_k=top_k, temperature=temperature)
        
        return {
            "success": True,
            "type": "qa",
            "answer": result.answer,
            "sources": result.sources,
            "query": result.query
        }
    
    def execute_stream(self, query: str, **kwargs):
        """流式问答"""
        top_k = kwargs.get("top_k", 5)
        
        # 先获取来源信息
        sources = self.rag_engine.get_relevant_context(query, top_k=top_k)
        
        # 流式生成
        for chunk in self.rag_engine.answer_with_stream(query, top_k=top_k):
            yield chunk


class CharacterQASkill(QASkill):
    """人物相关问答技能"""
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """执行人物相关问答"""
        # 优化人物相关问题的检索
        enhanced_query = f"人物 {query}"
        return super().execute(enhanced_query, **kwargs)


class ChapterQASkill(QASkill):
    """章节相关问答技能"""
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """执行章节相关问答"""
        # 优化章节相关问题的检索
        enhanced_query = f"章节 {query}"
        return super().execute(enhanced_query, **kwargs)
