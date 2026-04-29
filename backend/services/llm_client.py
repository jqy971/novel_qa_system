"""
阿里云大模型客户端 - 封装qwen-turbo API调用
"""
import requests
import json
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

from backend.config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, LLM_MODEL


@dataclass
class Message:
    """对话消息"""
    role: str  # system, user, assistant
    content: str


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    usage: Dict[str, int]
    finish_reason: str


class QwenClient:
    """阿里云通义千问客户端"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or DASHSCOPE_API_KEY
        self.model = model or LLM_MODEL
        self.base_url = DASHSCOPE_BASE_URL
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> LLMResponse:
        """
        非流式对话 - 使用兼容OpenAI格式的API
        
        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            temperature: 温度参数，控制创造性
            max_tokens: 最大生成token数
            stream: 是否流式返回
            
        Returns:
            LLM响应对象
        """
        # 使用兼容OpenAI的API端点
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            print(f"[API调用] 模型: {self.model}, 消息数: {len(messages)}")
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            # 调试：打印响应内容
            if response.status_code != 200:
                print(f"[Debug] Status: {response.status_code}")
                print(f"[Debug] Response: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            # 打印用量信息
            if "usage" in result:
                usage = result["usage"]
                print(f"[API用量] prompt_tokens: {usage.get('prompt_tokens', 0)}, completion_tokens: {usage.get('completion_tokens', 0)}, total: {usage.get('total_tokens', 0)}")
            
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                message = choice.get("message", {})
                
                return LLMResponse(
                    content=message.get("content", ""),
                    usage=result.get("usage", {}),
                    finish_reason=choice.get("finish_reason", "stop")
                )
            else:
                error_msg = result.get("error", {}).get("message", "未知错误")
                raise Exception(f"API返回异常: {error_msg}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API请求失败: {e}")
        except Exception as e:
            raise Exception(f"调用失败: {e}")
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Generator[str, None, None]:
        """
        流式对话 - 用于实时显示生成内容 - 使用兼容OpenAI格式
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大生成token数
            
        Yields:
            生成的文本片段
        """
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data:'):
                        try:
                            data = json.loads(line_text[5:])
                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]
                                delta = choice.get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            yield f"[错误: {e}]"
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: str = None,
        temperature: float = 0.7
    ) -> str:
        """
        基于上下文生成回答
        
        Args:
            query: 用户问题
            context: 检索到的上下文
            system_prompt: 系统提示词
            temperature: 温度参数
            
        Returns:
            生成的回答
        """
        if system_prompt is None:
            system_prompt = """你是一个专业的小说阅读助手。请严格基于提供的原文内容回答用户问题。
如果原文中没有相关信息，请明确告知用户。
回答时请保持客观，不要添加原文中没有的信息。"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""基于以下小说原文内容回答问题：

【原文内容】
{context}

【用户问题】
{query}

请根据原文内容回答，如果原文中没有相关信息，请说明。"""}
        ]
        
        response = self.chat(messages, temperature=temperature)
        return response.content
    
    def continue_story(
        self,
        context: str,
        style_sample: str,
        instruction: str = "",
        max_tokens: int = 1000
    ) -> str:
        """
        续写小说情节
        
        Args:
            context: 前文内容
            style_sample: 风格样本
            instruction: 续写指令
            max_tokens: 最大生成长度
            
        Returns:
            续写内容
        """
        system_prompt = """你是一个专业的小说续写助手。请严格遵循以下规则：
1. 仔细分析原文的风格、语气、人物设定
2. 续写内容必须与原文风格保持一致
3. 人物性格和行为必须符合原著设定
4. 不要偏离故事主线
5. 保持叙事的连贯性"""
        
        user_prompt = f"""请根据以下原文风格和前文，续写小说内容：

【风格样本】
{style_sample[:500]}...

【前文】
{context}

【续写要求】
{instruction if instruction else "请按照原文风格自然续写"}

请开始续写："""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.chat(messages, temperature=0.8, max_tokens=max_tokens)
        return response.content
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        生成文本摘要
        
        Args:
            text: 待摘要文本
            max_length: 摘要最大长度
            
        Returns:
            摘要内容
        """
        system_prompt = "你是一个专业的文本摘要助手。请用简洁的语言概括文本的主要内容。"
        
        user_prompt = f"""请对以下文本进行摘要，控制在{max_length}字以内：

{text[:3000]}

摘要："""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.chat(messages, temperature=0.5, max_tokens=300)
        return response.content


# 测试代码
if __name__ == "__main__":
    client = QwenClient()
    
    # 测试简单对话
    messages = [
        {"role": "system", "content": "你是一个 helpful 的助手。"},
        {"role": "user", "content": "你好，请介绍一下自己"}
    ]
    
    try:
        response = client.chat(messages)
        print(f"回答: {response.content}")
        print(f"Token使用: {response.usage}")
    except Exception as e:
        print(f"测试失败: {e}")
