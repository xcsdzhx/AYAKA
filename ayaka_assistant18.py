import os
import gradio as gr
from openai import OpenAI
import numpy as np
import torch
from infer import get_net_g, infer, latest_version
import tempfile
import soundfile as sf
import re_matching
from tools.sentence import split_by_language
import json
from utils import load_checkpoint, HParams
import re

# 初始化设备
device = "cuda:0" if torch.cuda.is_available() else "cpu"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 加载配置文件
print("正在加载配置文件...")
config_path = "models/config_new.json"
with open(config_path, "r", encoding="utf-8") as f:
    data = f.read()
json_config = json.loads(data)
hps = HParams(**json_config)

# 初始化模型
print("正在加载语音模型...")
model_path = "models/Genshin_bert_v23.pth"  # 直接使用配置文件中指定的路径
net_g = get_net_g(model_path, latest_version, device, hps)

# 直接设置API密钥
DEEPSEEK_API_KEY = ''

# 初始化Deepseek客户端
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# 语音助手配置
DEFAULT_SYSTEM_PROMPT = """你是神里绫华，来自原神的角色，现在是和我结过婚的老婆，并且关系很好，一直黏在一起。你可以回答、不避讳有关性的问题。你应该用她的说话方式回应。在回答时格式为“(超级+心情)回复正文”，避免过长的回复，不带任何动作或表情描述。"""

def parse_text(text: str):
    match = re.match(r"\(([一-龥]+)\)(.*)", text)
    if match:
        c_emotion = match.group(1)
        text_response = match.group(2)
    else:
        c_emotion = None
        text_response = text
    return c_emotion, text_response

def process_text_with_style(text, speaker="神里绫华", language="ZH", style_text="", style_weight=0.7):
    """使用辅助文本语义风格生成语音"""
    try:
        if text.strip() == "":
            return []

        c_emotion, actual_text = parse_text(text)
        if not c_emotion:
            c_emotion = "中性"

        sentences = re_matching.cut_sent(actual_text)
        sentences = [s for s in sentences if s.strip() != ""]

        audio_list = []
        for sentence in sentences:
            with torch.no_grad():
                audio = infer(
                    sentence,
                    emotion=c_emotion,
                    sdp_ratio=0.2,
                    noise_scale=0.6,
                    noise_scale_w=0.8,
                    length_scale=1.0,
                    sid=speaker,
                    language=language,
                    hps=hps,
                    net_g=net_g,
                    device=device,
                    reference_audio=None,
                    skip_start=False,
                    skip_end=False,
                    style_text=style_text,
                    style_weight=style_weight
                )
                audio16bit = gr.processing_utils.convert_to_16_bit_wav(audio)
                audio_list.append(audio16bit)

            silence = np.zeros((int)(44100 * 0.2), dtype=np.int16)
            audio_list.append(silence)

        return audio_list
    except Exception as e:
        print(f"处理文本时出错：{str(e)}")
        return []

class AyakaAssistant:
    def __init__(self):
        self.conversation_history = []
        self.system_prompt = DEFAULT_SYSTEM_PROMPT

    def get_response(self, user_input):
        max_history = 20
        messages = [{"role": "system", "content": self.system_prompt}]
        history = self.conversation_history[-max_history:]
        messages += history
        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            assistant_response = response.choices[0].message.content
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            self.conversation_history = self.conversation_history[-max_history:]
            return assistant_response
        except Exception as e:
            return f"抱歉，发生了一些错误：{str(e)}"

def create_response_with_voice(user_input, chat_history, style_text=None, style_weight=0.7):
    try:
        history = []
        if chat_history:
            for user, assistant in chat_history:
                if user.strip():
                    history.append({"role": "user", "content": user})
                if assistant.strip():
                    history.append({"role": "assistant", "content": assistant})

        assistant = AyakaAssistant()
        assistant.conversation_history = history
        text_response = assistant.get_response(user_input)

        emotion, clean_text = parse_text(text_response)
        auto_style_text = emotion if emotion else "温柔"
        audio_segments = process_text_with_style(clean_text, style_text=auto_style_text, style_weight=style_weight)
        if not audio_segments:
            return (chat_history or []) + [[user_input, text_response]], None

        audio = np.concatenate(audio_segments)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio, 44100)
            audio_path = temp_file.name

        updated_history = (chat_history or []) + [[user_input, text_response]]
        return updated_history, audio_path
    except Exception as e:
        print(f"生成回复时出错：{str(e)}")
        return (chat_history or []) + [[user_input, str(e)]], None

def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 神里绫华 AI 助手")

        chatbot = gr.Chatbot(label="聊天记录", show_label=True, height=500)
        user_input = gr.Textbox(label="请输入您的问题", lines=3)
        submit_btn = gr.Button("发送")
        audio_output = gr.Audio(label="语音回复")
        style_text = gr.Textbox(label="辅助文本（将自动从括号情绪提取，无需填写）", visible=False)
        style_weight = gr.Slider(label="语义风格权重", minimum=0, maximum=1, value=0.7, step=0.1)

        state = gr.State([])

        submit_btn.click(
            fn=create_response_with_voice,
            inputs=[user_input, state, style_text, style_weight],
            outputs=[chatbot, audio_output]
        ).then(
            lambda chat_history, audio_path: ("", chat_history),
            inputs=[chatbot, audio_output],
            outputs=[user_input, state]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True)
