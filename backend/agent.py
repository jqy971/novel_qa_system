"""
Agent 任务规划与调度
根据用户意图调度不同的Skill执行任务
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from backend.skills.intent_classifier import IntentClassifier, IntentType
from backend.skills.qa_skill import QASkill, CharacterQASkill, ChapterQASkill
from backend.skills.continue_skill import ContinueSkill
from backend.skills.extract_skill import ExtractSkill, SummarizeSkill


@dataclass
class Task:
    """任务定义"""
    id: str
    intent: str
    query: str
    novel_id: str
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = None


class NovelAgent:
    """
    小说阅读助手Agent
    
    职责：
    1. 识别用户意图
    2. 规划任务
    3. 调度Skill执行
    4. 返回结果
    """
    
    def __init__(self, novel_id: str):
        self.novel_id = novel_id
        self.intent_classifier = IntentClassifier()
        
        # 初始化Skills
        self.skills = {
            "qa": QASkill(novel_id),
            "character_qa": CharacterQASkill(novel_id),
            "chapter_qa": ChapterQASkill(novel_id),
            "continue": ContinueSkill(novel_id),
            "extract": ExtractSkill(novel_id),
            "summarize": SummarizeSkill(novel_id)
        }
        
        # 对话历史
        self.chat_history = []
    
    def process(self, query: str) -> Dict[str, Any]:
        """
        处理用户输入 - Agent核心调度逻辑
        
        Args:
            query: 用户输入
            
        Returns:
            处理结果
        """
        # 1. 意图识别
        intent_result = self.intent_classifier.classify_with_context(
            query, 
            self.chat_history
        )
        intent = intent_result["intent"]
        
        print(f"[Agent] 识别意图: {intent}, 查询: {query}")
        
        # 2. 任务路由 - 根据意图调度不同Skill
        try:
            if intent == IntentType.CONTINUE.value:
                result = self._handle_continue(query)
            elif intent == IntentType.SUMMARIZE.value:
                result = self._handle_summarize(query)
            elif intent == IntentType.CHARACTER.value:
                result = self._handle_character_qa(query)
            elif intent == IntentType.CHAPTER.value:
                result = self._handle_chapter_qa(query)
            elif intent == IntentType.GREETING.value:
                result = self._handle_greeting()
            else:
                # 默认问答
                result = self._handle_qa(query)
            
            # 3. 记录对话历史
            self.chat_history.append({
                "query": query,
                "intent": intent,
                "timestamp": datetime.now().isoformat()
            })
            
            # 限制历史长度
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
            return {
                "success": True,
                "intent": intent,
                "result": result
            }
            
        except Exception as e:
            print(f"[Agent] 处理失败: {e}")
            return {
                "success": False,
                "intent": intent,
                "error": str(e),
                "result": {"answer": f"处理出错: {str(e)}"}
            }
    
    def _handle_qa(self, query: str) -> Dict[str, Any]:
        """处理问答任务"""
        print(f"[Agent] 调度QASkill执行问答")
        return self.skills["qa"].execute(query)
    
    def _handle_character_qa(self, query: str) -> Dict[str, Any]:
        """处理人物相关问答"""
        print(f"[Agent] 调度CharacterQASkill执行人物问答")
        return self.skills["character_qa"].execute(query)
    
    def _handle_chapter_qa(self, query: str) -> Dict[str, Any]:
        """处理章节相关问答"""
        print(f"[Agent] 调度ChapterQASkill执行章节问答")
        return self.skills["chapter_qa"].execute(query)
    
    def _handle_continue(self, query: str) -> Dict[str, Any]:
        """处理续写任务"""
        print(f"[Agent] 调度ContinueSkill执行续写")
        return self.skills["continue"].execute(query)
    
    def _handle_summarize(self, query: str) -> Dict[str, Any]:
        """处理摘要任务"""
        print(f"[Agent] 调度SummarizeSkill执行摘要")
        
        # 尝试提取章节名
        import re
        chapter_match = re.search(r'第[一二三四五六七八九十百千万零\d]+章', query)
        chapter_title = chapter_match.group() if chapter_match else None
        
        return self.skills["summarize"].summarize_chapter(chapter_title)
    
    def _handle_greeting(self) -> Dict[str, Any]:
        """处理问候"""
        return {
            "type": "greeting",
            "answer": "你好！我是你的小说阅读助手。你可以问我关于这本小说的任何问题，或者让我帮你续写情节。"
        }
    
    def extract_characters(self) -> Dict[str, Any]:
        """抽取人物信息"""
        print(f"[Agent] 调度ExtractSkill抽取人物")
        return self.skills["extract"].extract_characters()
    
    def analyze_relationships(self, character_names: list) -> Dict[str, Any]:
        """分析人物关系"""
        print(f"[Agent] 调度ExtractSkill分析人物关系")
        return self.skills["extract"].analyze_relationships(character_names)


class AgentManager:
    """Agent管理器 - 管理多个小说的Agent实例"""
    
    _instances = {}
    
    @classmethod
    def get_agent(cls, novel_id: str) -> NovelAgent:
        """获取或创建Agent实例"""
        if novel_id not in cls._instances:
            cls._instances[novel_id] = NovelAgent(novel_id)
        return cls._instances[novel_id]
    
    @classmethod
    def remove_agent(cls, novel_id: str):
        """移除Agent实例"""
        if novel_id in cls._instances:
            del cls._instances[novel_id]


# 测试代码
if __name__ == "__main__":
    print("Agent模块已加载")
    print("使用方法: agent = NovelAgent('novel_id')")
    print("          result = agent.process('你的问题')")
