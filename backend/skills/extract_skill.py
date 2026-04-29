"""
信息抽取Skill - NLP任务4: 信息抽取
从小说中抽取人物、地点、事件等信息
"""
import re
from typing import List, Dict, Any, Set
from collections import Counter

from backend.services.vector_store import VectorStore
from backend.services.llm_client import QwenClient


class ExtractSkill:
    """信息抽取技能 - 人物、地点、关系抽取"""
    
    def __init__(self, novel_id: str):
        self.novel_id = novel_id
        self.vector_store = VectorStore(novel_id)
        self.llm_client = QwenClient()
    
    def extract_characters(self, sample_text: str = None) -> Dict[str, Any]:
        """
        抽取人物信息
        
        Args:
            sample_text: 样本文本，如果为None则从向量库抽样
            
        Returns:
            人物列表和统计
        """
        if sample_text is None:
            # 从向量库获取样本文本
            sample_text = self._get_sample_text()
        
        # 使用LLM抽取人物
        prompt = f"""请从以下小说文本中识别人物角色，并返回JSON格式结果：

文本内容：
{sample_text[:3000]}

请识别所有人物名称，并按以下JSON格式返回：
{{
    "characters": [
        {{"name": "人物名", "description": "简要描述", "importance": "主要/次要"}}
    ]
}}

只返回JSON，不要其他内容。"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的文学分析助手，擅长从小说中识别人物角色。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.3)
            content = response.content
            
            # 尝试解析JSON
            import json
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "success": True,
                    "type": "character_extraction",
                    "characters": result.get("characters", [])
                }
        except Exception as e:
            print(f"人物抽取失败: {e}")
        
        # Fallback：使用简单规则抽取
        characters = self._simple_character_extraction(sample_text)
        return {
            "success": True,
            "type": "character_extraction",
            "characters": characters,
            "method": "rule_based"
        }
    
    def _simple_character_extraction(self, text: str) -> List[Dict[str, Any]]:
        """基于规则的简单人物抽取"""
        # 常见姓氏
        surnames = ['萧', '林', '叶', '楚', '苏', '沈', '顾', '陆', '程', '李', '王', '张', '刘', '陈', '杨', '赵', '黄', '周', '吴', '徐', '孙', '胡', '朱', '高', '何', '郭', '马', '罗', '梁', '宋', '郑', '谢', '韩', '唐', '冯', '于', '董', '萧', '程', '曹', '袁', '邓', '许', '傅', '沈', '曾', '彭', '吕', '苏', '卢', '蒋', '蔡', '贾', '丁', '魏', '薛', '叶', '阎', '余', '潘', '杜', '戴', '夏', '钟', '汪', '田', '任', '姜', '范', '方', '石', '姚', '谭', '廖', '邹', '熊', '金', '陆', '郝', '孔', '白', '崔', '康', '毛', '邱', '秦', '江', '史', '顾', '侯', '邵', '孟', '龙', '万', '段', '雷', '钱', '汤', '尹', '黎', '易', '常', '武', '乔', '贺', '赖', '龚', '文']
        
        # 匹配2-4个字符的潜在人名
        pattern = r'[' + ''.join(surnames) + r'][\u4e00-\u9fa5]{1,3}(?=[，。！？、：""''（）\s]|$)'
        matches = re.findall(pattern, text)
        
        # 统计频率
        counter = Counter(matches)
        
        # 过滤低频词和常见非人名词
        stop_words = {'什么', '怎么', '这么', '那么', '这个', '那个', '一个', '没有', '不能', '可以', '就是', '还是', '但是', '因为', '所以', '如果', '虽然', '这些', '那些', '它们', '他们', '她们', '我们', '你们', '咱们'}
        
        characters = []
        for name, count in counter.most_common(30):
            if name not in stop_words and len(name) >= 2 and count >= 2:
                characters.append({
                    "name": name,
                    "description": f"在文中出现{count}次",
                    "importance": "主要" if count > 10 else "次要"
                })
        
        return characters[:20]
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        抽取关键词 - NLP任务：关键词抽取
        
        Args:
            text: 文本内容
            top_n: 返回关键词数量
            
        Returns:
            关键词列表
        """
        # 使用简单的TF-IDF思想
        # 这里简化处理，使用词频
        words = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)
        
        # 停用词
        stop_words = set(['什么', '怎么', '这么', '那么', '这个', '那个', '一个', '没有', '不能', '可以', '就是', '还是', '但是', '因为', '所以', '如果', '虽然', '这些', '那些', '它们', '他们', '她们', '我们', '你们', '咱们', '自己', '已经', '开始', '进行', '成为', '需要', '通过', '作为', '表示', '认为', '知道', '看到', '听到', '想到', '说道', '看着', '接着'])
        
        # 过滤并统计
        filtered_words = [w for w in words if w not in stop_words and len(w) >= 2]
        counter = Counter(filtered_words)
        
        return [word for word, _ in counter.most_common(top_n)]
    
    def analyze_relationships(self, character_names: List[str], sample_text: str = None) -> Dict[str, Any]:
        """
        分析人物关系
        
        Args:
            character_names: 人物名称列表
            sample_text: 样本文本
            
        Returns:
            人物关系分析结果
        """
        if sample_text is None:
            sample_text = self._get_sample_text()
        
        # 使用LLM分析关系
        prompt = f"""请分析以下人物之间的关系，并返回JSON格式结果：

人物列表：{', '.join(character_names[:10])}

文本样例：
{sample_text[:2000]}

请分析这些人物之间的关系，按以下格式返回：
{{
    "relationships": [
        {{"person1": "人物A", "person2": "人物B", "relation": "关系描述"}}
    ]
}}

只返回JSON，不要其他内容。"""
        
        messages = [
            {"role": "system", "content": "你是一个专业的文学分析助手，擅长分析小说中的人物关系。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.llm_client.chat(messages, temperature=0.3)
            content = response.content
            
            import json
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "success": True,
                    "type": "relationship_analysis",
                    "relationships": result.get("relationships", [])
                }
        except Exception as e:
            print(f"关系分析失败: {e}")
        
        return {
            "success": False,
            "type": "relationship_analysis",
            "relationships": [],
            "error": str(e)
        }
    
    def _get_sample_text(self, max_chunks: int = 10) -> str:
        """从向量库获取样本文本"""
        # 获取统计信息
        stats = self.vector_store.get_stats()
        total_chunks = stats.get("total_chunks", 0)
        
        if total_chunks == 0:
            return ""
        
        # 抽样检索
        sample_text = ""
        queries = ["主角", "故事", "情节", "人物"]
        for query in queries:
            results = self.vector_store.search(query, top_k=3)
            for r in results:
                sample_text += r["content"] + "\n"
        
        return sample_text[:5000]


class SummarizeSkill:
    """摘要技能"""
    
    def __init__(self, novel_id: str):
        self.novel_id = novel_id
        self.llm_client = QwenClient()
        self.vector_store = VectorStore(novel_id)
    
    def summarize_chapter(self, chapter_title: str = None) -> Dict[str, Any]:
        """
        生成章节摘要
        
        Args:
            chapter_title: 章节标题，如果为None则摘要全文
            
        Returns:
            摘要结果
        """
        # 获取章节内容
        if chapter_title:
            # 检索特定章节
            results = self.vector_store.search(chapter_title, top_k=10)
            chapter_content = "\n".join([r["content"] for r in results if chapter_title in r["metadata"].get("chapter_title", "")])
        else:
            # 获取全文样本
            results = self.vector_store.search("主要内容", top_k=10)
            chapter_content = "\n".join([r["content"] for r in results])
        
        if not chapter_content:
            return {
                "success": False,
                "type": "summarize",
                "summary": "未找到相关内容"
            }
        
        # 生成摘要
        summary = self.llm_client.summarize(chapter_content, max_length=300)
        
        # 抽取关键词
        keywords = self._extract_keywords(chapter_content)
        
        return {
            "success": True,
            "type": "summarize",
            "summary": summary,
            "keywords": keywords,
            "chapter": chapter_title or "全文"
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """抽取关键词"""
        words = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)
        stop_words = set(['什么', '怎么', '这么', '那么', '这个', '那个', '一个', '没有', '不能', '可以', '就是', '还是', '但是', '因为', '所以', '如果', '虽然', '这些', '那些'])
        filtered_words = [w for w in words if w not in stop_words and len(w) >= 2]
        counter = Counter(filtered_words)
        return [word for word, _ in counter.most_common(10)]
