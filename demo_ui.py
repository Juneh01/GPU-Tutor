"""
GPU学习智能体 - 现场演示界面

启动方式：
1. 先启动 serve.py（后端服务）
2. 再启动本文件（前端界面）

python demo_ui.py
"""

import gradio as gr
import requests
import time

# 后端服务地址
API_URL = "http://localhost:8000/predict"

def query_model(question: str) -> str:
    """调用后端API"""
    if not question.strip():
        return "请输入问题"
    
    start = time.time()
    
    try:
        response = requests.post(
            API_URL,
            json={"prompt": question},
            timeout=30
        )
        result = response.json()
        answer = result.get("response", "无响应")
        elapsed = time.time() - start
        
        return f"{answer}\n\n---\n⏱️ 响应时间: {elapsed:.2f}秒"
    
    except requests.exceptions.ConnectionError:
        return "❌ 连接失败，请确保serve.py已启动"
    except Exception as e:
        return f"❌ 错误: {str(e)}"

# 预设的演示问题
DEMO_QUESTIONS = [
    "什么是CUDA？",
    "共享内存(shared memory)如何优化GPU程序？",
    "什么是Warp divergence？如何避免？",
    "如何避免shared memory的bank conflict？",
    "GPU的SM包含哪些组件？",
    "CUDA如何实现矩阵乘法？",
    "什么是GPU的内存合并访问(coalesced access)？",
    "CUDA中如何实现线程同步？",
]

def select_demo_question(question):
    """选择预设问题"""
    return question

# 创建Gradio界面
with gr.Blocks(title="GPU学习智能体", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 GPU学习智能体
    ### 基于SFT+DPO微调与多索引RAG的高性能问答系统
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # 输入区域
            question_input = gr.Textbox(
                label="请输入GPU相关问题",
                placeholder="例如：什么是CUDA的shared memory？",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 提交问题", variant="primary")
                clear_btn = gr.Button("🗑️ 清空")
            
            # 输出区域
            answer_output = gr.Textbox(
                label="回答",
                lines=10,
                interactive=False
            )
        
        with gr.Column(scale=1):
            # 预设问题
            gr.Markdown("### 📝 预设问题（点击选择）")
            for q in DEMO_QUESTIONS:
                btn = gr.Button(q, size="sm")
                btn.click(fn=lambda x=q: x, outputs=question_input)
    
    # 绑定事件
    submit_btn.click(fn=query_model, inputs=question_input, outputs=answer_output)
    clear_btn.click(fn=lambda: ("", ""), outputs=[question_input, answer_output])
    
    # 回车提交
    question_input.submit(fn=query_model, inputs=question_input, outputs=answer_output)
    
    gr.Markdown("""
    ---
    **技术栈**: Qwen3-0.6B + SFT + DPO + 多索引RAG + vLLM
    """)

if __name__ == "__main__":
    print("="*50)
    print("GPU学习智能体 - 演示界面")
    print("="*50)
    print("\n请确保已启动后端服务: python serve.py")
    print("\n访问地址: http://localhost:7860")
    print("="*50)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # 如需公网访问设为True
    )
