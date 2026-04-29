"""
续写Skill - 封装续写能力
"""
from typing import Dict, Any
from backend.services.rag_engine import ContinueEngine


class ContinueSkill:
    """续写技能"""
    
    def __init__(self, novel_id: str):
        self.novel_id = novel_id
        self.continue_engine = ContinueEngine(novel_id)
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行续写
        
        Args:
            query: 续写指令
            **kwargs: 额外参数
            
        Returns:
            续写结果
        """
        context_length = kwargs.get("context_length", 1000)
        max_tokens = kwargs.get("max_tokens", 1000)
        
        result = self.continue_engine.continue_story(
            query=query,
            context_length=context_length,
            max_tokens=max_tokens
        )
        
        return {
            "success": True,
            "type": "continue",
            "continuation": result["continuation"],
            "sources": result["sources"],
            "query": query
        }
