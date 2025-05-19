import openai
from rich.console import Console

# === 配置你的 API 密钥 ===
openai.api_key = "你的API密钥"

# === 模型设置 ===
MODEL = "gpt-3.5-turbo"  # 或 "gpt-4"

# Rich 控制台输出，支持颜色和动画
console = Console()

# 对话历史记录
messages = [
    {"role": "system", "content": "你是一个友好且有帮助的AI助手。"}
]

def chat():
    console.print("[bold green]欢迎使用 ChatGPT 终端聊天机器人！输入 'exit' 退出。[/bold green]\n")

    while True:
        user_input = input("你：")
        if user_input.lower() in ["exit", "quit", "bye"]:
            console.print("[bold yellow]再见！[/bold yellow]")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            # 调用 OpenAI API
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=messages
            )

            reply = response.choices[0].message["content"]
            messages.append({"role": "assistant", "content": reply})

            console.print(f"[bold cyan]ChatGPT：[/bold cyan]{reply}\n")

        except Exception as e:
            console.print(f"[red]出错了：{e}[/red]")

if __name__ == "__main__":
    chat()
