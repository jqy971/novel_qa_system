"""
本地Embedding实现 - 无需API费用
基于TF-IDF和词频的轻量级向量表示
"""
import re
import math
from typing import List, Dict, Set
from collections import Counter
import numpy as np


class LocalEmbedding:
    """
    本地文本嵌入模型
    使用TF-IDF + 词频特征，无需外部API
    """
    
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.stopwords: Set[str] = self._load_stopwords()
        
    def _load_stopwords(self) -> Set[str]:
        """加载中文停用词"""
        # 基础停用词表
        stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那',
            '这些', '那些', '这个', '那个', '之', '与', '及', '等',
            '他', '她', '它', '他们', '她们', '它们', '得', '地',
            '而', '但', '如果', '因为', '所以', '虽然', '或者',
            '可以', '把', '被', '让', '向', '从', '为', '以',
            '将', '于', '即', '则', '乃', '且', '又', '并',
            '第', '章', '节', '回', '卷', '篇'
        }
        return stopwords
    
    def _tokenize(self, text: str) -> List[str]:
        """
        简单的中文分词
        基于字符和常见词模式
        """
        # 清理文本
        text = re.sub(r'[^\u4e00-\u9fff\w]', ' ', text)
        
        # 提取2-4字的词组
        words = []
        chars = list(text.replace(' ', ''))
        
        # 单字
        words.extend(chars)
        
        # 2-4字词组
        for n in range(2, 5):
            for i in range(len(chars) - n + 1):
                word = ''.join(chars[i:i+n])
                if len(word) >= 2:
                    words.append(word)
        
        # 过滤停用词和单字（保留有意义的单字）
        filtered = []
        for w in words:
            if len(w) == 1 and w in self.stopwords:
                continue
            if w in self.stopwords:
                continue
            filtered.append(w)
        
        return filtered
    
    def _compute_tf(self, words: List[str]) -> Dict[str, float]:
        """计算词频"""
        if not words:
            return {}
        counter = Counter(words)
        total = len(words)
        return {word: count/total for word, count in counter.items()}
    
    def fit(self, texts: List[str]):
        """
        从文本集合学习IDF
        
        Args:
            texts: 文本列表
        """
        # 收集所有文档的词
        doc_words = []
        for text in texts:
            words = set(self._tokenize(text))
            doc_words.append(words)
        
        # 计算IDF
        N = len(texts)
        all_words = set()
        for words in doc_words:
            all_words.update(words)
        
        for word in all_words:
            # 包含该词的文档数
            df = sum(1 for words in doc_words if word in words)
            # IDF = log(N / (df + 1)) + 1
            self.idf[word] = math.log(N / (df + 1)) + 1
        
        # 构建词汇表（选择最重要的词）
        sorted_words = sorted(
            self.idf.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.dim]
        
        self.vocab = {word: idx for idx, (word, _) in enumerate(sorted_words)}
        
    def embed(self, text: str) -> List[float]:
        """
        将文本转换为向量
        
        Args:
            text: 输入文本
            
        Returns:
            向量表示
        """
        words = self._tokenize(text)
        tf = self._compute_tf(words)
        
        # 构建向量
        vector = [0.0] * len(self.vocab)
        for word, tf_val in tf.items():
            if word in self.vocab:
                idx = self.vocab[word]
                idf_val = self.idf.get(word, 1.0)
                vector[idx] = tf_val * idf_val
        
        # 归一化
        norm = math.sqrt(sum(v*v for v in vector))
        if norm > 0:
            vector = [v/norm for v in vector]
        
        return vector
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入"""
        # 如果没有词汇表，先fit
        if not self.vocab:
            self.fit(texts)
        
        return [self.embed(text) for text in texts]


class SimpleEmbedding:
    """
    更简单的Embedding实现 - 基于字符哈希
    无需训练，即插即用
    """
    
    def __init__(self, dim: int = 256):
        self.dim = dim
        
    def _hash_vector(self, text: str) -> List[float]:
        """使用哈希将文本转换为向量"""
        vector = [0.0] * self.dim
        
        # 字符级特征
        for i, char in enumerate(text):
            # 使用字符的Unicode码点进行哈希
            char_code = ord(char) % self.dim
            # 位置敏感
            position_weight = 1.0 / (1 + i * 0.1)
            vector[char_code] += position_weight
            
            # 2-gram特征
            if i < len(text) - 1:
                bigram = text[i:i+2]
                bigram_hash = hash(bigram) % self.dim
                vector[bigram_hash] += position_weight * 1.5
        
        # 归一化
        norm = math.sqrt(sum(v*v for v in vector))
        if norm > 0:
            vector = [v/norm for v in vector]
        
        return vector
    
    def embed(self, text: str) -> List[float]:
        """嵌入单条文本"""
        return self._hash_vector(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入"""
        return [self.embed(text) for text in texts]


# 默认使用简单Embedding（无需训练）
EmbeddingModel = SimpleEmbedding


# 测试
if __name__ == "__main__":
    # 测试简单Embedding
    model = SimpleEmbedding(dim=256)
    
    texts = [
        "第一章 测试内容",
        "第二章 另一个测试",
        "第三章 完全不同的内容"
    ]
    
    embeddings = model.embed_batch(texts)
    print(f"生成的向量维度: {len(embeddings[0])}")
    
    # 计算相似度
    def cosine_sim(v1, v2):
        dot = sum(a*b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a*a for a in v1))
        norm2 = math.sqrt(sum(b*b for b in v2))
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    
    sim_0_1 = cosine_sim(embeddings[0], embeddings[1])
    sim_0_2 = cosine_sim(embeddings[0], embeddings[2])
    
    print(f"文本0和1的相似度: {sim_0_1:.4f}")
    print(f"文本0和2的相似度: {sim_0_2:.4f}")
