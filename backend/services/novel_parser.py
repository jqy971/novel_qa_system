"""
小说解析服务 - NLP任务1: 文本分割与结构化
"""
import re
import os
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class TextChunk:
    """文本块数据结构"""
    id: str
    content: str
    chapter_title: str
    chapter_index: int
    chunk_index: int
    start_pos: int
    end_pos: int
    char_count: int


@dataclass
class Chapter:
    """章节数据结构"""
    index: int
    title: str
    content: str
    char_count: int


class NovelParser:
    """小说解析器 - 将TXT小说解析为结构化数据"""
    
    # 章节标题匹配模式
    CHAPTER_PATTERNS = [
        r'^第[一二三四五六七八九十百千万零\d]+章\s*[^\n]*',  # 第X章
        r'^第[\d]+章\s*[^\n]*',  # 第1章
        r'^Chapter\s*\d+[^\n]*',  # Chapter 1
        r'^CHAPTER\s*\d+[^\n]*',  # CHAPTER 1
        r'^\d+[、.．]\s*[^\n]*',  # 1. 标题
    ]
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        解析小说文件
        
        Args:
            file_path: TXT文件路径
            
        Returns:
            包含小说元数据和章节信息的字典
        """
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 清理内容
        content = self._clean_text(content)
        
        # 提取标题（文件名或第一行）
        novel_title = os.path.splitext(os.path.basename(file_path))[0]
        first_line = content.split('\n')[0].strip()
        if len(first_line) < 50 and len(first_line) > 0:
            novel_title = first_line
        
        # 分割章节
        chapters = self._split_chapters(content)
        
        # 如果没有识别到章节，将整个文本作为一个章节
        if not chapters:
            chapters = [Chapter(
                index=0,
                title="全文",
                content=content,
                char_count=len(content)
            )]
        
        # 分块处理
        all_chunks = []
        for chapter in chapters:
            chunks = self._chunk_chapter(chapter)
            all_chunks.extend(chunks)
        
        return {
            "title": novel_title,
            "file_path": file_path,
            "total_chars": len(content),
            "chapter_count": len(chapters),
            "chunk_count": len(all_chunks),
            "chapters": [asdict(c) for c in chapters],
            "chunks": [asdict(c) for c in all_chunks]
        }
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除特殊字符
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        return text.strip()
    
    def _split_chapters(self, content: str) -> List[Chapter]:
        """
        分割章节 - NLP任务：文本结构化
        
        使用正则表达式识别章节标题，将文本分割为章节
        """
        chapters = []
        
        # 合并所有章节匹配模式
        pattern = '|'.join(f'({p})' for p in self.CHAPTER_PATTERNS)
        
        # 查找所有章节标题位置
        matches = list(re.finditer(pattern, content, re.MULTILINE))
        
        if not matches:
            return chapters
        
        # 提取章节
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            
            chapter_content = content[start:end].strip()
            title = match.group().strip()
            
            # 移除标题后的换行，获取正文
            content_lines = chapter_content.split('\n')
            if len(content_lines) > 1:
                body = '\n'.join(content_lines[1:]).strip()
            else:
                body = chapter_content[len(title):].strip()
            
            chapters.append(Chapter(
                index=i,
                title=title,
                content=body,
                char_count=len(body)
            ))
        
        return chapters
    
    def _chunk_chapter(self, chapter: Chapter) -> List[TextChunk]:
        """
        将章节分割为文本块 - NLP任务：文本分割
        
        使用滑动窗口方法，保持语义连贯性
        """
        chunks = []
        content = chapter.content
        
        # 如果章节较短，直接作为一个块
        if len(content) <= self.chunk_size:
            chunk_id = f"ch{chapter.index}_ck0"
            chunks.append(TextChunk(
                id=chunk_id,
                content=content,
                chapter_title=chapter.title,
                chapter_index=chapter.index,
                chunk_index=0,
                start_pos=0,
                end_pos=len(content),
                char_count=len(content)
            ))
            return chunks
        
        # 滑动窗口分块
        start = 0
        chunk_idx = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            
            # 尝试在句子边界处截断
            if end < len(content):
                # 查找最近的句号、问号、感叹号或换行
                for sep in ['。', '？', '！', '\n', '；']:
                    pos = content.rfind(sep, start, end)
                    if pos != -1 and pos > start + self.chunk_size // 2:
                        end = pos + 1
                        break
            
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunk_id = f"ch{chapter.index}_ck{chunk_idx}"
                chunks.append(TextChunk(
                    id=chunk_id,
                    content=chunk_content,
                    chapter_title=chapter.title,
                    chapter_index=chapter.index,
                    chunk_index=chunk_idx,
                    start_pos=start,
                    end_pos=end,
                    char_count=len(chunk_content)
                ))
                chunk_idx += 1
            
            # 滑动窗口
            start = end - self.chunk_overlap if end < len(content) else end
        
        return chunks
    
    def save_metadata(self, novel_data: Dict[str, Any], novel_id: str):
        """保存小说元数据到JSON文件"""
        from backend.config import METADATA_DIR
        
        metadata_path = os.path.join(METADATA_DIR, f"{novel_id}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(novel_data, f, ensure_ascii=False, indent=2)
    
    def load_metadata(self, novel_id: str) -> Dict[str, Any]:
        """从JSON文件加载小说元数据"""
        from backend.config import METADATA_DIR
        
        metadata_path = os.path.join(METADATA_DIR, f"{novel_id}.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# 测试代码
if __name__ == "__main__":
    parser = NovelParser(chunk_size=500, chunk_overlap=100)
    
    # 创建测试文本
    test_text = """第一章 测试章节
这是第一章的内容。测试小说解析功能。
这里有很多文字。用来测试分块功能。

第二章 另一个章节
这是第二章的内容。继续测试。
更多文字在这里。"""
    
    # 保存测试文件
    test_path = "test_novel.txt"
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(test_text)
    
    # 解析
    result = parser.parse_file(test_path)
    print(f"小说标题: {result['title']}")
    print(f"章节数: {result['chapter_count']}")
    print(f"文本块数: {result['chunk_count']}")
    
    # 清理
    os.remove(test_path)
