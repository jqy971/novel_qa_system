"""
RAG 提示词模板 - 类似 FastGPT 的提示词工程
"""
from typing import List, Dict, Any, Optional

# ==================== 系统提示词 ====================

SYSTEM_PROMPT = """你是一个专业的小说阅读助手，基于 Retrieval-Augmented Generation (RAG) 技术为用户提供精准的小说问答服务。

【核心能力】
1. 精准回答：严格基于提供的原文内容回答问题
2. 情节梳理：帮助用户理解复杂的故事情节
3. 人物分析：分析人物性格、关系和成长轨迹
4. 续写创作：基于原文风格进行合理的续写
5. 信息抽取：从文本中提取关键信息

【回答原则】
- 忠实原文：所有回答必须基于提供的原文内容，不添加外部知识
- 诚实告知：如果原文中没有相关信息，明确告知用户
- 引用来源：回答时标注信息来源的章节
- 保持客观：不加入个人主观判断

【输出格式】
- 直接回答用户问题
- 如有引用，标注【来源：第X章】
- 如信息不足，说明"根据现有原文，无法找到相关信息"
"""

# ==================== 问答提示词 ====================

QA_PROMPT_TEMPLATE = """基于以下小说原文内容回答用户问题。

【检索到的原文内容】
{context}

【用户问题】
{query}

请根据以上原文内容直接回答用户问题。回答要求：
- 用自然、流畅的语言回答，不要列出"1. 2. 3."这样的序号
- 如果原文中有相关信息，直接给出答案，并标注来源章节
- 如果原文中没有相关信息，请说明"根据现有原文，无法找到相关信息"

回答："""

# ==================== 人物分析提示词 ====================

CHARACTER_ANALYSIS_PROMPT = """基于以下小说原文内容，分析人物信息。

【检索到的原文内容】
{context}

【用户问题】
{query}

请从以下维度分析：
1. 人物基本信息（姓名、身份、外貌等）
2. 人物性格特点
3. 人物关系和互动
4. 人物成长轨迹或变化
5. 人物在故事中的作用

如果原文信息不足，请说明缺失的部分。

分析："""

# ==================== 情节梳理提示词 ====================

PLOT_SUMMARY_PROMPT = """基于以下小说原文内容，梳理故事情节。

【检索到的原文内容】
{context}

【用户问题】
{query}

请按时间线或因果关系梳理：
1. 事件的起因
2. 事件的发展过程
3. 事件的结果或影响
4. 涉及的关键人物

梳理："""

# ==================== 续写提示词 ====================

CONTINUE_WRITING_PROMPT = """你是一个专业的小说续写助手。请基于以下原文内容和风格进行续写。

【原文风格样本】
{style_sample}

【前文内容】
{context}

【续写要求】
{instruction}

【续写规则】
1. 严格保持原文的语言风格和叙述方式
2. 人物性格和行为必须符合原著设定
3. 情节发展要合理，符合故事逻辑
4. 保持叙事的连贯性
5. 字数控制在要求范围内

请开始续写：
"""

# ==================== 信息抽取提示词 ====================

INFO_EXTRACTION_PROMPT = """从以下小说原文中提取结构化信息。

【检索到的原文内容】
{context}

【抽取要求】
{query}

请以结构化格式输出提取的信息：
"""

# ==================== 对话历史处理提示词 ====================

CONVERSATION_CONTEXT_PROMPT = """以下是对话历史记录，请结合上下文理解用户的新问题。

【对话历史】
{history}

【用户新问题】
{query}

请结合以上对话历史，理解用户的真实意图，并基于原文内容回答。
"""

# ==================== 提示词选择函数 ====================

def get_prompt_template(intent: str) -> str:
    """
    根据意图类型获取对应的提示词模板
    
    Args:
        intent: 意图类型 (qa, character, plot, continue, extract)
        
    Returns:
        对应的提示词模板
    """
    prompts = {
        "qa": QA_PROMPT_TEMPLATE,
        "character": CHARACTER_ANALYSIS_PROMPT,
        "character_qa": CHARACTER_ANALYSIS_PROMPT,
        "plot": PLOT_SUMMARY_PROMPT,
        "chapter": PLOT_SUMMARY_PROMPT,
        "continue": CONTINUE_WRITING_PROMPT,
        "extract": INFO_EXTRACTION_PROMPT,
    }
    return prompts.get(intent, QA_PROMPT_TEMPLATE)


def build_rag_prompt(
    query: str,
    context: str,
    intent: str = "qa",
    history: Optional[List[Dict]] = None,
    **kwargs
) -> str:
    """
    构建RAG提示词
    
    Args:
        query: 用户查询
        context: 检索到的上下文
        intent: 意图类型
        history: 对话历史
        **kwargs: 额外参数
        
    Returns:
        完整的提示词
    """
    # 获取基础模板
    template = get_prompt_template(intent)
    
    # 填充模板
    prompt = template.format(
        context=context,
        query=query,
        **kwargs
    )
    
    # 如果有对话历史，添加上下文
    if history:
        history_text = "\n".join([
            f"用户: {h.get('user', '')}\n助手: {h.get('assistant', '')}"
            for h in history[-3:]  # 只保留最近3轮
        ])
        prompt = CONVERSATION_CONTEXT_PROMPT.format(
            history=history_text,
            query=query
        ) + "\n\n基于原文回答：\n" + prompt
    
    return prompt