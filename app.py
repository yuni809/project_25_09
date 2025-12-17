import streamlit as st
import time
import json
import os
import uuid 
from datetime import datetime
import base64
import re
import numpy as np 

try:
    import main as backend
except ImportError:
    st.error("Error: main.py not found.")
    st.stop()

# --- 설정 및 CSS ---
st.set_page_config(page_title="EmoDiary", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    .stApp { background-image: linear-gradient(to bottom, #e3f2fd, #ffffff); color: black; }
    h1, h2, h3, p, div, span, label { color: black !important; font-family: 'Helvetica Neue', sans-serif; text-align: center; }
    div.stButton > button { border-radius: 30px; background-color: #4285F4 !important; color: white !important; border: none; padding: 8px 18px; width: 100%; }
    .glass-card { background: rgba(255, 255, 255, 0.9); border-radius: 15px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 15px; text-align: center; }
    .history-card { background: rgba(255, 255, 255, 0.95); border-radius: 15px; padding: 15px; margin-bottom: 12px; }
    .stChatInput textarea { background-color: #ffffff !important; color: #000000 !important; border: 1px solid #cccccc !important; }
    audio { display: none !important; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = r"C:\Users\Owner\Desktop\SCI & 박사논문\SCI"
HISTORY_FILE = "emotion_history.json"
GIF_IDLE = os.path.join(BASE_DIR, "idle.gif")
GIF_LISTEN = os.path.join(BASE_DIR, "listening.gif")

EMO_IMAGES = {
    "happy": os.path.join(BASE_DIR, "happy.png"),
    "sad": os.path.join(BASE_DIR, "sad.png"),
    "neutral": os.path.join(BASE_DIR, "neutral.png"),
    "angry": os.path.join(BASE_DIR, "anger.png"),
    "fear": os.path.join(BASE_DIR, "fear.png"),
    "disgust": os.path.join(BASE_DIR, "disgust.png"),
    "surprise": os.path.join(BASE_DIR, "surprised.png")
}
DEFAULT_IMG = os.path.join(BASE_DIR, "neutral.png")

EMOTION_KO = {"happy": "행복", "sad": "슬픔", "neutral": "중립", "angry": "분노", "fear": "공포", "disgust": "혐오", "surprise": "놀람"}
VIDEO_FEAT_KO = {
    "browDownLeft": "왼쪽 눈썹 내림", "browDownRight": "오른쪽 눈썹 내림",
    "mouthFrownLeft": "왼쪽 입꼬리 내림", "mouthFrownRight": "오른쪽 입꼬리 내림",
    "mouthSmileLeft": "왼쪽 입꼬리 올림", "mouthSmileRight": "오른쪽 입꼬리 올림"
}

def get_base64_image(file_path):
    if not os.path.exists(file_path): return ""
    with open(file_path, "rb") as f: data = f.read()
    return base64.b64encode(data).decode()

def autoplay_audio_hidden(file_path):
    if not os.path.exists(file_path): return
    with open(file_path, "rb") as f:
        data = f.read()
    st.audio(data, format="audio/mp3", autoplay=True)

def local_record_audio_only(output_wav="chat_voice.wav", duration=5, fs=16000):
    try:
        recording = backend.sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        backend.sd.wait()
        backend.wav.write(output_wav, fs, recording)
        return output_wav
    except: return None

# --- State ---
if 'step' not in st.session_state: st.session_state['step'] = 'intro'
if 'language' not in st.session_state: st.session_state['language'] = 'ko'
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []
if 'user_emotion' not in st.session_state: st.session_state['user_emotion'] = "neutral"
if 'intro_played' not in st.session_state: st.session_state['intro_played'] = False
if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None
if 'last_tts' not in st.session_state: st.session_state['last_tts'] = None

# ==========================================
# 1. INTRO
# ==========================================
if st.session_state['step'] == 'intro':
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        lang_choice = st.radio("Language", ["한국어", "English"], horizontal=True, label_visibility="collapsed")
        st.session_state['language'] = 'ko' if lang_choice == "한국어" else 'en'
    with c3:
        if st.button("📖 일기 레포트", use_container_width=True):
            st.session_state['step'] = 'history'; st.rerun()

    if os.path.exists(GIF_IDLE):
        b64 = get_base64_image(GIF_IDLE)
        st.markdown(f"""<div style="display: flex; justify-content: center; margin: 20px;"><img src="data:image/gif;base64,{b64}" width="350" style="border-radius:15px;"></div>""", unsafe_allow_html=True)
    
    greeting = "오늘 무슨 일이 있었나요? 편하게 말씀해 주세요." if st.session_state['language'] == 'ko' else "How was your day?"
    st.markdown(f"<h3 style='margin-bottom: 30px;'>{greeting}</h3>", unsafe_allow_html=True)

    if not st.session_state['intro_played']:
        tts_path = backend.text_to_speech(greeting, "intro_tts.mp3")
        if tts_path: autoplay_audio_hidden(tts_path); st.session_state['last_tts'] = tts_path
        st.session_state['intro_played'] = True

    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        if st.button("🎙️ 대답하기 (녹화 시작)", use_container_width=True):
            st.session_state['step'] = 'recording'; st.rerun()

# ==========================================
# 2. RECORDING
# ==========================================
elif st.session_state['step'] == 'recording':
    lang = st.session_state['language']
    
    if os.path.exists(GIF_LISTEN):
        b64 = get_base64_image(GIF_LISTEN)
        st.markdown(f"""<div style="display: flex; justify-content: center; margin: 20px;"><img src="data:image/gif;base64,{b64}" width="350" style="border-radius:15px;"></div>""", unsafe_allow_html=True)
    
    msg = "듣고 있어요... 말씀해 주세요." if lang=='ko' else "Listening... Please speak."
    st.markdown(f"<h3>{msg}</h3>", unsafe_allow_html=True)

    with st.spinner("Recording..."):
        video_path = backend.record_realtime_multimodal(output_path="user_input.mp4", duration=10)
    st.session_state['video_path'] = video_path
    st.session_state['step'] = 'analysis_preview'
    st.rerun()

# ==========================================
# 3. ANALYSIS PREVIEW
# ==========================================
elif st.session_state['step'] == 'analysis_preview':
    lang = st.session_state['language']
    
    title = "🔍 감정 분석 결과" if lang=='ko' else "Analysis Result"
    sub = "AI가 당신의 음성, 텍스트, 표정을 분석했습니다." if lang=='ko' else "AI analyzed your voice, text, and face."
    st.markdown(f"<h2>{title}</h2><p>{sub}</p>", unsafe_allow_html=True)
    
    if st.session_state.get('analysis_result') is None:
        with st.spinner("Analyzing..."):
            result = backend.run_full_pipeline(st.session_state['video_path'], language=lang)
            st.session_state['analysis_result'] = result

    res = st.session_state['analysis_result']
    details = res['details']

    speech_data = details.get('speech', {})
    speech_hyp_dict = speech_data.get('hypothesis', {})
    audio_hyp = max(speech_hyp_dict, key=speech_hyp_dict.get) if (speech_hyp_dict and 'error' not in speech_hyp_dict) else "Unknown"
    audio_evi_dict = speech_data.get('evidence', {})
    if audio_evi_dict:
        audio_evi_str = f"음높이: {audio_evi_dict.get('pitch_hz', 0)}Hz, 음량: {audio_evi_dict.get('vol_db', 0)}dB" if lang=='ko' else f"Pitch: {audio_evi_dict.get('pitch_hz', 0)}Hz"
    else: audio_evi_str = "N/A"

    text_data = details.get('text', {})
    text_hyp = text_data.get('hypothesis', {}).get('emotion', 'neutral')
    text_evi_str = text_data.get('evidence', {}).get('context', 'N/A')

    video_data = details.get('video', {})
    video_hyp_dict = video_data.get('hypothesis', {})
    video_hyp = max(video_hyp_dict, key=video_hyp_dict.get) if (video_hyp_dict and 'error' not in video_hyp_dict) else "neutral"
    video_evi_dict = video_data.get('evidence', {})
    if video_evi_dict:
        sorted_feats = sorted(video_evi_dict.items(), key=lambda x: x[1], reverse=True)[:2]
        video_evi_str = ", ".join([f"{VIDEO_FEAT_KO.get(k, k) if lang=='ko' else k}: {v:.2f}" for k, v in sorted_feats])
    else: video_evi_str = "N/A"

    if lang == 'ko':
        audio_hyp_ko = EMOTION_KO.get(audio_hyp, audio_hyp)
        text_hyp_ko = EMOTION_KO.get(text_hyp, text_hyp)
        video_hyp_ko = EMOTION_KO.get(str(video_hyp), str(video_hyp))
    else:
        audio_hyp_ko = audio_hyp; text_hyp_ko = text_hyp; video_hyp_ko = video_hyp

    c1, c2, c3 = st.columns(3)
    with c1: st.info(f"**🎤 음성**\n\n**가설:** {audio_hyp_ko.upper()}\n\n**증거:** {audio_evi_str}")
    with c2: st.info(f"**📝 텍스트**\n\n**가설:** {text_hyp_ko.upper()}\n\n**증거:** {text_evi_str}")
    with c3: st.info(f"**📹 영상**\n\n**가설:** {video_hyp_ko.upper()}\n\n**증거:** {video_evi_str}")

    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        btn_txt = "💬 대화 시작하기" if lang=='ko' else "Start Chat"
        if st.button(btn_txt, use_container_width=True):
            final_judgment = res['final_judgment']
            st.session_state['user_emotion'] = final_judgment.get('final_emotion', 'neutral')
            st.session_state['analysis_rationale'] = final_judgment.get('rationale', '')
            
            sys_prompt = f"User Emotion: {st.session_state['user_emotion']}. Context: {res['transcript']}. Start conversation warmly in Korean."
            gen_res = backend.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": sys_prompt}])
            first_msg = gen_res.choices[0].message.content
            
            st.session_state['chat_history'] = [{"role": "assistant", "content": first_msg}]
            unique_filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
            st.session_state['last_tts'] = backend.text_to_speech(first_msg, unique_filename)
            st.session_state['step'] = 'chatting'; st.rerun()

# ==========================================
# 4. CHATTING
# ==========================================
elif st.session_state['step'] == 'chatting':
    lang = st.session_state['language']
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists(GIF_IDLE):
            b64 = get_base64_image(GIF_IDLE)
            st.markdown(f"""<div class="avatar-container"><img src="data:image/gif;base64,{b64}" class="avatar-img" style="max-width: 380px;"></div>""", unsafe_allow_html=True)
        if st.session_state.get('last_tts'):
            autoplay_audio_hidden(st.session_state['last_tts']); st.session_state['last_tts'] = None

    with st.container(height=300):
        for msg in st.session_state['chat_history']:
            with st.chat_message(msg['role']): st.write(msg['content'])

    user_input = st.chat_input("메시지 입력..." if lang=='ko' else "Type message...")
    c1, c2 = st.columns([2, 5])
    with c1:
        if st.button("🎤 음성으로 말하기", use_container_width=True):
            with st.spinner("Listening..."):
                temp_wav = local_record_audio_only()
                if temp_wav:
                    try: 
                        wl = "ko" if lang == "ko" else "en"
                        res = backend.whisper_model.transcribe(temp_wav, language=wl, initial_prompt="일상 대화, 감정 표현")
                        user_input = res.get("text", "")
                    except: pass
    
    if user_input:
        st.session_state['chat_history'].append({"role": "user", "content": user_input})
        ctx = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state['chat_history']])
        sys_p = f"Friend, Emotion:{st.session_state['user_emotion']}, Lang:{'Korean' if lang=='ko' else 'English'}"
        res = backend.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":sys_p},{"role":"user","content":ctx}])
        reply = res.choices[0].message.content
        st.session_state['chat_history'].append({"role": "assistant", "content": reply})
        unique_filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"
        st.session_state['last_tts'] = backend.text_to_speech(reply, unique_filename)
        st.rerun()

    if st.button("⏹ 상담 종료" if lang=='ko' else "Finish", use_container_width=True):
        st.session_state['step'] = 'report'; st.rerun()

# ==========================================
# 5. REPORT 
# ==========================================
elif st.session_state['step'] == 'report':
    lang = st.session_state['language']
    emotion = st.session_state['user_emotion']
    full_chat = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state['chat_history']])
    
    with st.spinner("Generating Report..."):
        rag_advice = backend.get_rag_advice(emotion, full_chat)
        
        if lang == 'ko':
            prompt = f"""
            당신은 전문 심리 상담가입니다.
            
            [대화 내용]
            {full_chat}
            
            [참고 자료(전문 지식)]
            {rag_advice}
            
            [작성 규칙]
            1. 절대 번호(1., 2.)나 리스트 형식을 사용하지 마세요.
            2. 줄글(Paragraph) 형태로 자연스럽게 작성하세요.
            3. 아래 형식을 정확히 지켜주세요.

            요약: [대화 내용과 사용자 상황을 2~3문장으로 자연스럽게 요약]
            
            조언: [감정({emotion})에 공감하며, 위 참고 자료를 바탕으로 한 구체적이고 따뜻한 해결책 1~2문단]
            """
        else:
            prompt = f"""
            Summarize chat and advise for {emotion}.
            Reference: {rag_advice}
            
            [Rules]
            1. Do NOT use numbered lists. Use natural paragraphs.
            2. Format strictly as below:
            
            Summary: [Text]
            
            Advice: [Text]
            """
            
        res = backend.client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user", "content":prompt}])
        final_text = res.choices[0].message.content
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        disp_emotion = EMOTION_KO.get(emotion, emotion) if lang == 'ko' else emotion
        st.markdown(f"<div class='glass-card'><h2>Final Emotion: {disp_emotion.upper()}</h2></div>", unsafe_allow_html=True)
        
        img_key = emotion.lower()
        if 'happ' in img_key: img_key = 'happy'
        elif 'sad' in img_key: img_key = 'sad'
        elif 'ang' in img_key: img_key = 'angry'
        elif 'fear' in img_key: img_key = 'fear'
        elif 'disgust' in img_key: img_key = 'disgust'
        elif 'surprise' in img_key: img_key = 'surprise'
        elif 'neutral' in img_key: img_key = 'neutral'
        
        img_path = EMO_IMAGES.get(img_key, DEFAULT_IMG)
        if os.path.exists(img_path):
            b64 = get_base64_image(img_path)
            st.markdown(f"""<div style="display:flex; justify-content:center; margin-bottom:20px;">
                <img src='data:image/png;base64,{b64}' width='200' style="border-radius:15px;"></div>""", unsafe_allow_html=True)
        
        st.markdown(f"<div class='glass-card' style='text-align:left; white-space: pre-wrap;'>{final_text}</div>", unsafe_allow_html=True)
        
        save_txt = "저장하고 홈으로" if lang=='ko' else "Save & Home"
        if st.button(save_txt):
            new_record = {"date": datetime.now().strftime("%Y.%m.%d %H:%M"), "emotion": emotion, "summary": final_text, "timestamp": time.time()}
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f: h = json.load(f)
            except: h = []
            h.append(new_record)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f: json.dump(h, f, indent=4, ensure_ascii=False)
            st.session_state['step'] = 'intro'; st.session_state['intro_played'] = False; st.session_state['analysis_result'] = None; st.session_state['chat_history'] = []; st.rerun()

# ==========================================
# 6. HISTORY
# ==========================================
elif st.session_state['step'] == 'history':
    st.markdown("<h2>📒 감정 일기장</h2>", unsafe_allow_html=True)
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f: data = json.load(f)
        for item in sorted(data, key=lambda x:x['timestamp'], reverse=True):
            raw_emo = item['emotion'].lower()
            final_key = raw_emo
            if 'happ' in raw_emo: final_key = 'happy'
            elif 'sad' in raw_emo: final_key = 'sad'
            elif 'ang' in raw_emo: final_key = 'angry'
            elif 'fear' in raw_emo: final_key = 'fear'
            elif 'disgust' in raw_emo: final_key = 'disgust'
            elif 'surprise' in raw_emo: final_key = 'surprise'
            elif 'neutral' in raw_emo: final_key = 'neutral'

            img_path = EMO_IMAGES.get(final_key, DEFAULT_IMG)
            img_html = ""
            if os.path.exists(img_path):
                b64 = get_base64_image(img_path)
                img_html = (f"<img src='data:image/png;base64,{b64}' width='80' style='border-radius:50%; border:2px solid #ddd;'>")
            
            display_emo = EMOTION_KO.get(final_key, final_key.upper())
            st.markdown(f"""
            <div class="history-card"><div style="display:flex; align-items:center;">
                <div style="flex:1; text-align:center;">{img_html}</div>
                <div style="flex:4; padding-left:20px;">
                    <h3 style="margin:0; font-size:1.2em; text-align:left;">{item['date']} - {display_emo}</h3>
                    <p style="margin:5px 0 0 0; font-size:0.95em; color:#555; text-align:left; white-space: pre-wrap;">{item.get('summary', '요약 없음')}</p>
                </div>
            </div></div>""", unsafe_allow_html=True)
    else: st.info("기록 없음")
    if st.button("🏠 홈으로"): st.session_state['step'] = 'intro'; st.rerun()