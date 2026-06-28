import azure.cognitiveservices.speech as speechsdk

speech_config = speechsdk.SpeechConfig(
    subscription="lpimHJHXPdVGYVMsSmRoSz4huYXknP2NvouRBJ4gxKAksrJZzlFeJQQJ99BBAC3pKaRXJ3w3AAAYACOGIiAR",
    region="eastasia"
)
speech_config.speech_synthesis_voice_name = "zh-TW-HsiaoChenNeural"
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff48Khz16BitMonoPcm)

# 不使用播放功能
audio_config = None
speech_synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config,
    audio_config=audio_config
)

def synthesize_speech(text: str, emotion: str) -> bytes:
    rate = "+3%"
    pitch = "+0.35st"
    volume = "+0%"

    if emotion == "positive":
        rate = "+6%"
        pitch = "+0.45st"
        volume = "+0%"
    elif emotion == "negative":
        rate = "0%"
        pitch = "+0.25st"
        volume = "-5%"

    ssml_text = f"""
    <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='zh-TW'>
        <voice name='zh-TW-HsiaoChenNeural'>
            <prosody rate='{rate}' pitch='{pitch}' volume='{volume}'>
                {text}
            </prosody>
        </voice>
    </speak>
    """

    try:
        result = speech_synthesizer.speak_ssml_async(ssml_text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:

            # 建立音訊串流
            stream = speechsdk.AudioDataStream(result)

            # 存成 wav
            stream.save_to_wav_file("output.wav")

            # 讀檔案內容當作 byte 傳送給 Unity
            with open("output.wav", "rb") as f:
                wav_data = f.read()
                return wav_data

        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            print(f"❌ 合成失敗: {cancellation.reason}, 詳細錯誤資訊: {cancellation.error_details}")
            return None

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
        return None