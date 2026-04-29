"""
意图识别Skill - NLP任务3: 意图识别
识别用户是想问答、续写还是其他操作
"""
import re
from typing import Dict, Any
from enum import Enum


class IntentType(Enum):
    """意图类型"""
    QA = "qa"                    # 问答
    CONTINUE = "continue"        # 续写
    SUMMARIZE = "summarize"      # 摘要
    CHARACTER = "character"      # 人物相关
    CHAPTER = "chapter"          # 章节相关
    GREETING = "greeting"        # 问候
    UNKNOWN = "unknown"          # 未知


class IntentClassifier:
    """意图分类器 - 基于规则和关键词匹配"""
    
    # 意图关键词模式
    INTENT_PATTERNS = {
        IntentType.CONTINUE: [
            r'续写', r'接着写', r'往下写', r'后面.*怎么样', r'之后.*发生',
            r'如果.*会怎样', r'假设.*', r'改写', r'重写', r'续集',
            r'continue', r'write.*next', r'what.*happened.*after'
        ],
        IntentType.SUMMARIZE: [
            r'总结', r'摘要', r'概括', r'主要内容', r'讲了什么',
            r'summary', r'summarize', r'overview'
        ],
        IntentType.CHARACTER: [
            r'谁', r'人物', r'角色', r'主角', r'配角', r'关系',
            r'character', r'who', r'protagonist'
        ],
        IntentType.CHAPTER: [
            r'第.*章', r'章节', r'哪一章', r'chapter', r'哪一集'
        ],
        IntentType.GREETING: [
            r'^你好', r'^您好', r'^嗨', r'^hello', r'^hi$', r'^在吗',
            r'^你是谁', r'^你是什么', r'^你能做什么', r'^介绍一下',
            r'^你是', r'^请介绍', r'^what are you', r'^who are you'
        ]
    }
    
    def __init__(self):
        self.compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        识别用户意图
        
        Args:
            query: 用户输入
            
        Returns:
            包含意图类型和置信度的字典
        """
        query = query.strip()
        
        # 检查各种意图模式
        for intent_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    return {
                        "intent": intent_type.value,
                        "confidence": 0.9,
                        "matched_pattern": pattern.pattern
                    }
        
        # 默认分类为问答
        return {
            "intent": IntentType.QA.value,
            "confidence": 0.7,
            "matched_pattern": None
        }
    
    def classify_with_context(
        self,
        query: str,
        chat_history: list = None
    ) -> Dict[str, Any]:
        """
        结合上下文的意图识别
        
        Args:
            query: 用户输入
            chat_history: 对话历史
            
        Returns:
            意图识别结果
        """
        result = self.classify(query)
        
        # 如果有对话历史，考虑上下文
        if chat_history and len(chat_history) > 0:
            last_intent = chat_history[-1].get("intent", "qa")
            
            # 如果用户输入很短，可能是延续上一个意图
            if len(query) < 10 and last_intent in ["continue", "qa"]:
                result["intent"] = last_intent
                result["confidence"] = 0.6
                result["from_context"] = True
        
        return result


# 测试
if __name__ == "__main__":
    classifier = IntentClassifier()
    
    test_queries = [
        "萧炎后来怎么样了？",
        "帮我续写这段剧情",
        "总结一下第一章",
        "主角是谁？",
        "你好",
        "第三章讲了什么"
    ]
    
    for query in test_queries:
        result = classifier.classify(query)
        print(f"'{query}' -> {result}")
