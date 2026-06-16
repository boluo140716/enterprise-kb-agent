"""
Gradio Web 启动入口：从 web 包导入 demo 并启动服务
"""
from web import demo

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
    )
