import os
from pathlib import Path
import subprocess
import json
import time
import threading
import numpy as np
import cv2
import sounddevice as sd
import scipy.io.wavfile as wav
import torch
import torchaudio
import librosa
import mediapipe as mp
import whisper
from openai import OpenAI
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from collections import Counter

# ==========================================
# 1. 설정 및 경로
# ==========================================

BASE_PATH = Path(".").resolve()

# [경로 설정]
WAV2VEC_MODEL_DIR = Path(r"C:\Users\Owner\Desktop\SCI & 박사논문\SCI\70_accuracy_final_model")
MEDIAPIPE_FILENAME = "face_landmarker.task"
MEDIAPIPE_MODEL_PATH = BASE_PATH / MEDIAPIPE_FILENAME
VECTOR_DB_PATH = Path("dbt_vector_db") # RAG 데이터베이스 폴더

# FFMPEG 설정
FFMPEG_BIN = Path(r"C:\Users\Owner\ffmpeg\bin\ffmpeg.exe")
os.environ["PATH"] = str(FFMPEG_BIN.parent) + os.pathsep + os.environ["PATH"]

# API Key
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MEDIAPIPE_MODEL_PATH):
    print(f"▶ MediaPipe 모델 다운로드...")
    try:
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MEDIAPIPE_FILENAME)
    except Exception as e: print(f"!! 모델 다운로드 실패: {e}")

# ==========================================
# 2. 모델 로드 (RAG 포함)
# ==========================================
print("▶ Loading Models...")

# 1. RAG DB 로드
def load_rag_db():
    if not os.path.exists(VECTOR_DB_PATH):
        print("!! RAG DB 폴더를 찾을 수 없습니다. (일반 조언 모드로 동작)")
        return None
    try:
        print("   - RAG DB 로딩 중...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local(
            str(VECTOR_DB_PATH), 
            embeddings, 
            index_name="dbt_index", 
            allow_dangerous_deserialization=True
        )
        print("   ✅ RAG DB 로드 완료!")
        return db
    except Exception as e:
        print(f"!! RAG DB 로드 실패: {e}")
        return None

RAG_DB = load_rag_db()

# 2. Wav2Vec2
try:
    if os.path.exists(WAV2VEC_MODEL_DIR):
        speech_feature_extractor = AutoFeatureExtractor.from_pretrained(str(WAV2VEC_MODEL_DIR))
        speech_model = Wav2Vec2ForSequenceClassification.from_pretrained(str(WAV2VEC_MODEL_DIR)).to(device)
    else:
        speech_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        speech_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h").to(device)
except: speech_model = None

# 3. Whisper (Small 모델)
try:
    print("   - Whisper(Small) 로드 중...")
    whisper_model = whisper.load_model("small")
except Exception as e:
    print("!! Whisper Error:", e); whisper_model = None

# ==========================================
# 3. 유틸리티 & RAG 검색
# ==========================================
def text_to_speech(text, output_file="output_tts.mp3"):
    try:
        response = client.audio.speech.create(model="tts-1", voice="nova", input=text)
        response.stream_to_file(output_file)
        return output_file
    except: return None

def get_rag_advice(emotion, text_context):
    """RAG DB에서 조언 검색"""
    if RAG_DB is None: return ""
    try:
        query = f"Emotion: {emotion}, Situation: {text_context}. Therapy skills or advice."
        docs = RAG_DB.similarity_search(query, k=2)
        return "\n".join([f"- {d.page_content}" for d in docs])
    except Exception as e:
        print(f"RAG Error: {e}")
        return ""

def record_realtime_multimodal(output_path="live_test.mp4", duration=5):
    fs = 44100; audio_recording = []
    def record_audio():
        try:
            rec = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait(); audio_recording.append(rec)
        except: pass

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return None
    cap.set(3, 640); cap.set(4, 480)
    out = cv2.VideoWriter("temp_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))

    t = threading.Thread(target=record_audio); t.start()
    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret: break
        rem = int(duration - (time.time() - start_time))
        cv2.putText(frame, f"REC: {rem}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); out.release(); cv2.destroyAllWindows(); t.join()

    if not audio_recording: return None
    wav.write("temp_audio.wav", fs, audio_recording[0])
    
    if os.path.exists(output_path): os.remove(output_path)
    subprocess.call(f'"{str(FFMPEG_BIN)}" -y -i "temp_video.mp4" -i "temp_audio.wav" -c:v copy -c:a aac "{output_path}" -loglevel quiet', shell=True)
    
    if os.path.exists("temp_video.mp4"): os.remove("temp_video.mp4")
    if os.path.exists("temp_audio.wav"): os.remove("temp_audio.wav")
    return output_path

def extract_audio_from_video(video_path, out_wav_path):
    subprocess.run([str(FFMPEG_BIN), "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==========================================
# 4. 분석 엔진
# ==========================================

def analyze_speech_hypothesis(audio_path):
    if speech_model is None: return {}
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        intervals = librosa.effects.split(y, top_db=25)
        y_clean = np.concatenate([y[s:e] for s,e in intervals]) if len(intervals)>0 else y
        if len(y_clean) < 0.5*sr: y_clean = y
        
        inputs = speech_feature_extractor(y_clean, sampling_rate=16000, return_tensors="pt", padding=True, truncation=True, max_length=16000*10).to(device)
        with torch.no_grad(): logits = speech_model(inputs.input_values).logits
        scores = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        id2label = speech_model.config.id2label
        return {id2label[i]: round(float(s), 4) for i,s in enumerate(scores)}
    except: return {}

def analyze_speech_evidence(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitch = np.nanmean(f0[voiced_flag]) if np.any(voiced_flag) else 0
        rms = librosa.feature.rms(y=y); vol = librosa.amplitude_to_db(np.mean(rms))
        if hasattr(librosa.feature, 'rhythm'): tempo = librosa.feature.rhythm.tempo(y=y, sr=sr)[0]
        else: tempo = librosa.beat.tempo(onset_envelope=librosa.onset.onset_detect(y=y, sr=sr), sr=sr)[0]
        return {"pitch_hz": round(float(pitch), 2), "vol_db": round(float(vol), 2), "bpm": round(float(tempo), 2)}
    except: return {"pitch_hz": 0, "vol_db": 0, "bpm": 0}

def analyze_text_with_gpt(text, language="ko"):
    lang_inst = "Korean" if language == "ko" else "English"
    emotions = "['happy', 'sad', 'neutral', 'angry', 'fear', 'surprise', 'disgust']"
    prompt = f"""
    Analyze text: "{text}"
    [Rules]
    1. Hypothesis: Choose ONE emotion STRICTLY from {emotions}.
    2. Evidence: Explain context briefly in {lang_inst}.
    Output JSON: {{ "hypothesis": {{ "emotion": "LABEL", "confidence": 0.9 }}, "evidence": {{ "context": "DESC" }} }}
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role":"user", "content":prompt}], response_format={"type":"json_object"}
        )
        return json.loads(res.choices[0].message.content)
    except: return {"hypothesis": {}, "evidence": {}}

def analyze_video_hypothesis(video_path):
    try: from deepface import DeepFace
    except: return {}
    cap = cv2.VideoCapture(video_path)
    score = {k:0.0 for k in ["angry","disgust","fear","happy","sad","surprise","neutral"]}
    cnt = 0; valid = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        cnt += 1
        if cnt % 10 == 0:
            try:
                res = DeepFace.analyze(frame, actions=['emotion'], detector_backend='opencv', enforce_detection=False, silent=True)
                if res and isinstance(res[0], dict):
                    valid += 1
                    for e, v in res[0]['emotion'].items():
                        if e.lower() in score: score[e.lower()] += v
            except: pass
    cap.release()
    if valid == 0: return {"neutral": 1.0}
    score['neutral'] *= 0.5; tot = sum(score.values())
    return {k: round(v/tot, 4) for k, v in score.items()} if tot > 0 else {"neutral": 1.0}

def analyze_video_evidence(video_path):
    model_abs_path = os.path.abspath(MEDIAPIPE_FILENAME)
    if not os.path.exists(model_abs_path): return {}
    try:
        with open(model_abs_path, 'rb') as f: content = f.read()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=content),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, output_face_blendshapes=True
        )
    except: return {}

    INTERESTING = ["browDownLeft", "browDownRight", "mouthFrownLeft", "mouthFrownRight", "mouthSmileLeft", "mouthSmileRight"]
    scores = {n: [] for n in INTERESTING}
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                res = landmarker.detect_for_video(mp_img, ts)
                if res.face_blendshapes:
                    for b in res.face_blendshapes[0]:
                        if b.category_name in INTERESTING: scores[b.category_name].append(b.score)
            cap.release()
        return {n: round(np.mean(s), 4) if s else 0.0 for n,s in scores.items()}
    except: return {}

# ==========================================
# 5. 파이프라인
# ==========================================
def run_full_pipeline(video_path, language="ko"):
    temp_wav = BASE_PATH / "temp_web.wav"
    extract_audio_from_video(str(video_path), str(temp_wav))

    # STT
    transcript = ""
    if whisper_model:
        prompt_hint = "감정 분석, 일상 대화, 기분, 고민 상담, 후련하다, 속상하다, 기쁘다, 행복하다"
        try:
            res = whisper_model.transcribe(str(temp_wav), language=language, initial_prompt=prompt_hint)
            transcript = res.get("text", "").strip()
        except: transcript = ""

    speech_hyp = analyze_speech_hypothesis(str(temp_wav))
    speech_evi = analyze_speech_evidence(str(temp_wav))
    
    text_res = analyze_text_with_gpt(transcript, language=language)
    text_hyp = text_res.get("hypothesis")
    text_evi = text_res.get("evidence")
    
    video_hyp = analyze_video_hypothesis(str(video_path))
    video_evi = analyze_video_evidence(str(video_path))

    lang_txt = "Korean" if language == "ko" else "English"
    final_prompt = f"""
    You are a Master Multimodal Analyst.
    [Reports]
    Speech: {json.dumps(speech_hyp)} (Warning: Audio often misinterprets loud excitement as anger. Trust Text more.)
    Evidence: {json.dumps(speech_evi)}
    Text: {json.dumps(text_hyp)} (★ CRITICAL: Most accurate indicator)
    Context: {json.dumps(text_evi)}
    Video: {json.dumps(video_hyp)}
    
    [Transcript] "{transcript}"

    [Task]
    Determine final emotion from ["happy", "sad", "neutral", "angry", "fear", "disgust", "surprise"].
    * Rule 1: PRIORITIZE Text & Context over Audio noise.
    * Rule 2: IF Text is positive BUT Audio is angry -> Final is HAPPY.
    Output JSON: "final_emotion", "rationale" (in {lang_txt}).
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role":"system","content":"JSON only"},{"role":"user","content":final_prompt}], response_format={"type":"json_object"}
        )
        final_judgment = json.loads(res.choices[0].message.content)
    except Exception as e:
        final_judgment = {"final_emotion": "error", "rationale": str(e)}

    return {
        "transcript": transcript,
        "final_judgment": final_judgment,
        "details": {
            "speech": {"hypothesis": speech_hyp, "evidence": speech_evi},
            "text": {"hypothesis": text_hyp, "evidence": text_evi},
            "video": {"hypothesis": video_hyp, "evidence": video_evi}
        }
    }

if __name__ == "__main__":
    print("Module Loaded.")