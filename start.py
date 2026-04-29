"""
启动脚本 - 一键启动智能辅助阅读系统
"""
import os
import sys
import subprocess
import webbrowser
import time


def check_dependencies():
    """检查依赖是否安装"""
    # 设置环境变量禁用ChromaDB的默认embedding
    os.environ["CHROMA_DISABLE_TELEMETRY"] = "true"
    os.environ["ANONYMIZED_TELEMETRY"] = "false"
    
    try:
        import fastapi
        import requests
        print("✓ 核心依赖检查通过")
        
        # 尝试导入chromadb，但允许失败
        try:
            import chromadb
            print("✓ ChromaDB已安装")
        except Exception as e:
            print(f"⚠ ChromaDB导入警告: {e}")
            print("  尝试继续启动...")
        
        return True
    except ImportError as e:
        print(f"✗ 缺少核心依赖: {e}")
        print("请手动安装: pip install fastapi uvicorn requests")
        return False


def start_backend():
    """启动后端服务"""
    print("\n🚀 启动后端服务...")
    print("   API地址: http://localhost:8000")
    print("   文档地址: http://localhost:8000/docs\n")
    
    # 使用subprocess启动uvicorn
    backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
    
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    return process


def open_frontend():
    """打开前端页面"""
    time.sleep(2)  # 等待后端启动
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "index.html")
    
    if os.path.exists(frontend_path):
        print(f"🌐 正在打开前端页面...")
        webbrowser.open(f"file:///{frontend_path}")
    else:
        print("✗ 前端页面不存在")


def main():
    """主函数"""
    print("=" * 50)
    print("📚 小说智能问答系统 启动器")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("依赖安装失败，请手动安装: pip install -r requirements.txt")
        return
    
    # 检查API Key
    from backend.config import DASHSCOPE_API_KEY
    if DASHSCOPE_API_KEY == "your-api-key-here":
        print("\n⚠️ 警告: 请在 backend/config.py 中设置你的阿里云API Key")
        print("   获取地址: https://dashscope.aliyun.com/")
        print("   设置后重新启动\n")
        input("按回车键继续...")
        return
    
    # 启动后端
    backend_process = start_backend()
    
    # 打开前端
    open_frontend()
    
    print("\n✅ 系统已启动！")
    print("-" * 50)
    print("使用说明：")
    print("1. 在浏览器中上传TXT格式小说")
    print("2. 等待处理完成（自动分块、生成向量）")
    print("3. 开始提问或要求续写")
    print("-" * 50)
    print("按 Ctrl+C 关闭服务\n")
    
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 正在关闭服务...")
        backend_process.terminate()


if __name__ == "__main__":
    main()
