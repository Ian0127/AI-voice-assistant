import azure.cognitiveservices.speech as speechsdk
import sounddevice as sd

from scipy.io.wavfile import write
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

speech_config = speechsdk.SpeechConfig(subscription="lpimHJHXPdVGYVMsSmRoSz4huYXknP2NvouRBJ4gxKAksrJZzlFeJQQJ99BBAC3pKaRXJ3w3AAAYACOGIiAR", region="eastasia")
speech_config.speech_recognition_language = "zh-TW"
audio_config = speechsdk.AudioConfig(use_default_microphone=True)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# 載入 Transformers 的 pipeline 模型
classifier = pipeline("audio-classification", model="superb/hubert-large-superb-er")

# 錄音並存成wav檔案
# DURATION : 錄音時間（秒）
def record_wav(filename="input.wav", duration=3, samplerate=16000):
    print("🎙 開始錄音...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    write(filename, samplerate, audio)
    print(f"✅ 錄音完成，已儲存為 {filename}")

# Azure 語音辨識 
def recognize_from_wav(filename="input.wav"):
    """從 wav 檔進行語音辨識"""
    audio_config = speechsdk.AudioConfig(filename=filename)

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("開始辨識...")
    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"🔹 辨識結果：{result.text}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("⚠️ 無法辨識語音內容。")
        return None

# 語音情緒辨識函式
def analyze_emotion(AUDIO_FILE="input.wav"):
    print("\n 分析情緒中...")
    results = classifier(AUDIO_FILE)
    # 轉成完整標籤 & 四捨五入
    label_map = {
        'hap': 'happy',
        'sad': 'sad',
        'ang': 'angry',
        'neu': 'neutral'
    }
    max_label = None
    max_score = 0.0
    
    print(" 情緒辨識結果：")
    for r in results:
        label = label_map.get(r['label'], r['label'])
        score = round(r['score'], 3)
        print(f"{label}: {score}")
        if score > max_score:
            max_label = label
            max_score = score
    
    print(f" 主要情緒：{max_label}（{max_score:.3f}）")

    #  回傳最高分的情緒與分數
    return max_label

# 主函式
def recognize_speech() -> str:

    record_wav("input.wav", duration=3)
    result = recognize_from_wav("input.wav")
    max_emotion = analyze_emotion("input.wav")
    
    return result , max_emotion
    
# 測試程式
if __name__ == "__main__":
#    record_wav("input.wav", duration=5)
#    recognize_from_wav("input.wav")
#    analyze_emotion("input.wav")
    
    emotion = "Sad"
    
    # emotion_test = "emotion_test_data/" + emotion.lower() + "/" + emotion.lower() + "38.wav"
    emotion_test = "Emotion Speech Dataset/0003/" + emotion + "/0003_001240.wav"
    

    # recognize_from_wav(emotion_test)
    analyze_emotion(emotion_test)