import streamlit as st
import requests
import time
import os
from openai import OpenAI

# 自定义CSS样式 - 匹配HTML深色主题
st.set_page_config(
    page_title="情智星球 · 语句情感分析",
    page_icon="💭",
    layout="centered"
)

# 注入自定义CSS
custom_css = """
<style>
/* 全局背景与字体 */
.stApp {
    background-color: #0a0a1a;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif;
}

/* 主标题样式 */
.main-title {
    text-align: center;
    padding: 40px 20px 30px;
    margin-bottom: 20px;
}

.main-title h1 {
    font-size: 48px !important;
    font-weight: 700;
    background: linear-gradient(135deg, #fff 0%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 16px;
}

.main-title .subtitle {
    font-size: 16px;
    color: rgba(255, 255, 255, 0.7);
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.7;
}

/* 输入框容器 */
.input-container {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 32px;
    margin: 30px 0;
    transition: all 0.3s ease;
}

.input-container:hover {
    background: rgba(255, 255, 255, 0.07);
    border-color: rgba(167, 139, 250, 0.2);
}

/* 文本输入框 */
.stTextArea textarea {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #000000 !important;
    font-size: 15px !important;
    padding: 16px !important;
    min-height: 120px !important;
}

.stTextArea textarea:focus {
    border-color: rgba(167, 139, 250, 0.4) !important;
    box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.1) !important;
}

.stTextArea label {
    color: rgba(255, 255, 255, 0.7) !important;
    font-size: 15px !important;
    font-weight: 500;
}

/* 按钮样式 */
.stButton button {
    background: linear-gradient(135deg, #c084fc 0%, #a855f7 100%) !important;
    border: none !important;
    border-radius: 999px !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    width: 100% !important;
    transition: opacity 0.2s, transform 0.2s !important;
}

.stButton button:hover {
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(168, 85, 247, 0.3) !important;
}

/* 成功提示框 */
.stSuccess {
    background: rgba(52, 211, 153, 0.1) !important;
    border: 1px solid rgba(52, 211, 153, 0.3) !important;
    border-radius: 12px !important;
    color: #34d399 !important;
    padding: 16px 20px !important;
}

/* 信息提示框 */
.stInfo {
    background: rgba(167, 139, 250, 0.08) !important;
    border: 1px solid rgba(167, 139, 250, 0.2) !important;
    border-radius: 12px !important;
    color: rgba(255, 255, 255, 0.8) !important;
    padding: 12px 20px !important;
    font-size: 14px !important;
}

/* 错误提示框 */
.stError {
    background: rgba(248, 113, 113, 0.1) !important;
    border: 1px solid rgba(248, 113, 113, 0.3) !important;
    border-radius: 12px !important;
    color: #f87171 !important;
    padding: 16px 20px !important;
}

/* 底部说明文字 */
.footer-note {
    text-align: center;
    color: rgba(255, 255, 255, 0.45);
    font-size: 13px;
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

/* 隐藏Streamlit默认元素 */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* 结果展示卡片 */
.result-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-top: 20px;
}

/* 聊天消息容器 */
.stChatMessage {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
}

/* AI助手消息样式 */
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%) !important;
}

/* 用户消息样式 */
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%) !important;
}

/* 聊天输入框 */
.stChatInput {
    background: rgba(10, 10, 26, 0.95) !important;
    border: none !important;
}

[data-testid="stChatInput"] {
    background: transparent !important;
}

[data-testid="stChatInputTextAreaContainer"] {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 8px !important;
}

.stChatInput textarea {
    background-color: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(167, 139, 250, 0.2) !important;
    border-radius: 12px !important;
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 15px !important;
    padding: 14px 18px !important;
    min-height: 50px !important;
}

.stChatInput textarea:focus {
    background-color: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(167, 139, 250, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.1) !important;
    outline: none !important;
}

.stChatInput textarea::placeholder {
    color: rgba(255, 255, 255, 0.4) !important;
}

/* 发送按钮 */
[data-testid="stChatSubmitButton"] {
    background: linear-gradient(135deg, #c084fc 0%, #a855f7 100%) !important;
    border: none !important;
}

/* 底部容器和所有相关元素 */
[data-testid="stBottomBlockContainer"] {
    background: rgba(10, 10, 26, 0.98) !important;
}

[data-testid="stVerticalBlock"] {
    background: transparent !important;
}

.st-emotion-cache-1ab1jlb,
.st-emotion-cache-6mn6c9,
.st-emotion-cache-1wp6n47,
.st-emotion-cache-dx87ye,
.st-emotion-cache-137ctyu,
.st-emotion-cache-1p0pcv1,
.st-emotion-cache-1g8seck {
    background: transparent !important;
}

/* 禁用状态的按钮 */
button:disabled {
    background: rgba(167, 139, 250, 0.3) !important;
    opacity: 0.5 !important;
}

/* 最外层容器 - 覆盖所有白色背景 */
.main .block-container {
    background: #0a0a1a !important;
}

[data-testid="stAppViewContainer"] {
    background: #0a0a1a !important;
}

.st-emotion-cache-128upt6,
.st-emotion-cache-6shykm,
.st-emotion-cache-tn0cau,
.st-emotion-cache-1vo6xi6,
.st-emotion-cache-e1rw0b1u3,
.st-emotion-cache-e1rw0b1u1 {
    background: transparent !important;
}

/* 聊天区域标题 */
.chat-section-title {
    text-align: center;
    padding: 20px;
    margin: 30px 0 20px;
    background: rgba(167, 139, 250, 0.08);
    border-radius: 12px;
    border: 1px solid rgba(167, 139, 250, 0.15);
}

.chat-section-title h2 {
    font-size: 24px !important;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 8px;
}

.chat-section-title p {
    font-size: 14px;
    color: rgba(255, 255, 255, 0.6);
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 情感类别中英文映射
EMOTION_TRANSLATION = {
    'anger': '愤怒',
    'fear': '恐惧',
    'joy': '喜悦',
    'love': '爱',
    'sadness': '悲伤',
    'surprise': '惊讶'
}

def translate_emotion(pred_class):
    """将英文情感类别翻译为中文"""
    return EMOTION_TRANSLATION.get(pred_class, pred_class)

def translate_to_english(text):
    """使用DeepSeek API将中文文本翻译为英文"""
    try:
        client = init_deepseek_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个专业的翻译助手。请将用户的中文文本准确翻译为英文，保持原意不变。只返回翻译结果，不要添加任何解释或其他内容。"},
                {"role": "user", "content": text}
            ],
            stream=False
        )
        translated_text = response.choices[0].message.content.strip()
        return translated_text
    except Exception as e:
        # 如果翻译失败，返回原文
        st.warning(f"⚠️ 翻译失败，使用原文进行预测：{str(e)}")
        return text

# 初始化DeepSeek客户端
@st.cache_resource
def init_deepseek_client():
    """初始化DeepSeek API客户端"""
    return OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com"
    )

# 初始化会话状态
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pred_class' not in st.session_state:
    st.session_state.pred_class = None
if 'pred_class_cn' not in st.session_state:
    st.session_state.pred_class_cn = None

# 页面头部
st.markdown("""
<div class="main-title">
    <h1>语句情感智能分析</h1>
    <p class="subtitle">基于BERT深度学习模型，精准识别文本情感倾向，助力舆情分析与用户洞察</p>
</div>
""", unsafe_allow_html=True)

# 输入区域
text_input = st.text_area("请输入要分析的语句：", 
                          value="今天天气真好，心情特别愉快！",
                          height=120)

# 预测按钮
if st.button("开始预测"):
    if not text_input.strip():
        st.error("请输入要分析的语句！")
    else:
        # 显示翻译进度
        with st.spinner("正在翻译文本..."):
            # 将中文翻译为英文
            english_text = translate_to_english(text_input)
        
        # 构造请求体（使用翻译后的英文文本）
        data = {'text': english_text}
        url = 'http://localhost:8003/predict'
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            response = requests.post(url, json=data)
            # 预测消耗时间（包含翻译时间）
            use_time = (time.time() - start_time) * 1000
            
            # 请求成功
            if response.status_code == 200:
                result = response.json()
                pred_class = result["pred_class"]
                pred_class_cn = translate_emotion(pred_class)
                
                # 保存到会话状态
                st.session_state.pred_class = pred_class
                st.session_state.pred_class_cn = pred_class_cn
                
                # 显示结果卡片
                st.markdown(f"""
                <div class="result-card">
                    <div style="color: rgba(255, 255, 255, 0.7); font-size: 14px; margin-bottom: 8px;">BERT模型情感分析结果</div>
                    <div style="font-size: 24px; font-weight: 700; color: #a78bfa; margin-bottom: 16px;">
                        {pred_class_cn}
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px; color: rgba(255, 255, 255, 0.6); font-size: 13px;">
                        <span style="width: 6px; height: 6px; border-radius: 50%; background: #a78bfa;"></span>
                        BERT预测耗时：{use_time:.2f}ms
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 初始化聊天历史
                st.session_state.chat_history = [
                    {"role": "system", "content": f"你是一个专业的情感分析助手。当前语句被识别为【{pred_class_cn}】情感。请基于这个情感倾向与用户进行对话，解答用户关于情感分析的问题，提供情感洞察和建议。"}
                ]
                
                st.success("✅ 分析完成！现在可以与AI助手就这个情感倾向进行对话了")
            
            # 其它状态码请求失败
            else:
                st.error(f'请求失败：{response.json()["error"]}')
        
        except Exception as e:
            st.error(f'请求出错：{str(e)}')

# 如果有预测结果，显示聊天界面
if st.session_state.pred_class:
    # 美化聊天区域标题
    st.markdown("""
    <div class="chat-section-title">
        <h2>🤖 AI助手对话</h2>
        <p>基于BERT预测的情感倾向，为您提供专业的分析和解答</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示聊天历史
    for message in st.session_state.chat_history:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(f'<div style="line-height: 1.8; color: rgba(255, 255, 255, 0.9);">{message["content"]}</div>', unsafe_allow_html=True)
    
    # 用户输入
    user_input = st.chat_input("请输入您的问题...")
    
    if user_input:
        # 添加用户消息到历史
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(f'<div style="line-height: 1.8; color: rgba(255, 255, 255, 0.95);">{user_input}</div>', unsafe_allow_html=True)
        
        # 调用DeepSeek API
        with st.chat_message("assistant"):
            with st.spinner("🤔 AI正在思考中..."):
                try:
                    client = init_deepseek_client()
                    
                    deepseek_response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=st.session_state.chat_history,
                        stream=False
                    )
                    
                    assistant_response = deepseek_response.choices[0].message.content
                    
                    # 显示并保存助手回复
                    st.markdown(f'<div style="line-height: 1.8; color: rgba(255, 255, 255, 0.9); white-space: pre-wrap;">{assistant_response}</div>', unsafe_allow_html=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    
                except Exception as e:
                    error_msg = f"⚠️ DeepSeek调用失败：{str(e)}"
                    st.warning(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# 底部说明
st.markdown("""
<div class="footer-note">
    <p>© 2026 情智星球 · 请确保Flask API服务已在 localhost:8003 运行</p>
</div>
""", unsafe_allow_html=True)
