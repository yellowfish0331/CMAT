#!/usr/bin/env python3
"""
测试headless渲染环境是否正常工作
"""
import os
import sys

def test_environment():
    print("🧪 测试headless渲染环境...")
    
    # 检查DISPLAY
    display = os.environ.get('DISPLAY')
    print(f"📺 DISPLAY: {display}")
    if not display:
        print("❌ DISPLAY环境变量未设置")
        return False
    
    # 检查X11库
    try:
        import subprocess
        result = subprocess.run(['python', '-c', 'import tkinter; print("tkinter可用")'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ X11/tkinter环境正常")
        else:
            print(f"⚠️ X11/tkinter测试失败: {result.stderr}")
    except Exception as e:
        print(f"⚠️ X11测试异常: {e}")
    
    # 检查OpenGL
    try:
        result = subprocess.run(['python', '-c', 
                               'import moderngl; ctx = moderngl.create_context(standalone=True); print("OpenGL可用")'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ OpenGL环境正常")
        else:
            print(f"⚠️ OpenGL测试失败: {result.stderr}")
    except Exception as e:
        print(f"⚠️ OpenGL测试异常: {e}")
    
    # 检查OmniGibson导入
    try:
        import omnigibson
        print(f"✅ OmniGibson已安装: {omnigibson.__file__}")
    except ImportError as e:
        print(f"❌ OmniGibson导入失败: {e}")
        return False
    
    print("🎉 环境测试完成")
    return True

if __name__ == "__main__":
    test_environment()
