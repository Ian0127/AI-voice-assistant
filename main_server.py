import socket, asyncio, aiohttp, json, time, re
from LLM_Handler import ChatGPTResponseHandler, ChatHistoryManager
from stt_module import recognize_speech
from emotion_module import StreamEmotionAnalyzer
from tts_module import synthesize_speech
HOST = "127.0.0.1"
PORT = 65432
end_punctuation = set("。！？!?")

user_emotion = ""

def send_tts(sentense: str, emotion: str, conn: socket.socket) -> None:
    """執行語音合成"""
    sentense = sentense.strip()
    if not sentense:
        raise ValueError("空句子，跳過語音合成")
    try:
        audio_bytes = synthesize_speech(sentense, emotion)
        if not isinstance(audio_bytes, bytes):
            raise ValueError("語音合成結果不是 bytes 類型")
        if not audio_bytes:
            raise ValueError("語音合成結果為空")
        print(f"開始傳送文句: {sentense} (情緒: {emotion})")
        conn.sendall(f"\nEmotion:{emotion}\n".encode("utf-8"))
        time.sleep(0.1)
        conn.sendall(f"\nTTS_AUDIO:{len(audio_bytes)}\n".encode("utf-8"))
        time.sleep(0.1)
        conn.sendall(audio_bytes)
        return
    except Exception as e:
        print(f"TTS 錯誤: {e}")
        conn.sendall(f"Emotion:neutral".encode("utf-8"))
        conn.sendall(f"TTS:語音合成錯誤".encode("utf-8"))

async def update_weather_info(LLM: ChatGPTResponseHandler, conn: socket.socket) -> None:
    """更新天氣資訊"""
    print("開始更新天氣資訊...")
    auth_code = "CWA-E5891066-7F94-4F3F-8831-2E4EAB7AE670"
    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-C0032-001"
    location = "雲林縣"  # 查詢的縣市名稱
    params = {
        "Authorization": auth_code,
        "locationName": location,
        "elementName": "MinT,MaxT,Wx", # 查詢的天氣元素
        "timeFrom": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),  # 當前時間
        "timeTo": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 86400)),  # 24hr後的時間
    }

    try:
        async with aiohttp.ClientSession() as session:
            # 使用非同步的 get 方法，程式會在等待回應時切換到其他任務
            async with session.get(url, params=params) as response:
                # 檢查回應狀態碼，如果不是 200，會拋出 HTTPError
                response.raise_for_status()
                # 使用 await 讀取非同步回應，並解析為 JSON
                data = await response.json()
        
        # 提取所需資料
        weather_elements = data["records"]["location"][0]["weatherElement"]
        avg_temp = 0
        icon_str = None
        for element in weather_elements:
            # TODO: 若需求只要固定資料，可以直接取索引
            element_name = element["elementName"]
            measure_time = element["time"][0]
            parameter_name = measure_time["parameter"]["parameterName"]
            if element_name == "Wx":
                print(parameter_name)
                if "晴" in parameter_name:
                    icon_str = "sunny"
                if "雲" in parameter_name:
                    icon_str = "cloudy"
                if "陰" in parameter_name or "雨" in parameter_name:
                    icon_str = "rainy"
                if "轉" in parameter_name or "短暫" in parameter_name:
                    icon_str = "cloudy"
                if icon_str is None:
                    raise ValueError(f"無法解析天氣圖示，預報回傳值為{parameter_name}")
            if element_name == "MinT":
                avg_temp += int(parameter_name)
            if element_name == "MaxT":
                avg_temp += int(parameter_name)
        conn.sendall(f"\nCOMMAND:weather:{avg_temp//2}:{icon_str}\n".encode("utf-8"))
        LLM.OtherParams["currentTemp"] = avg_temp // 2
        LLM.OtherParams["currentWeather"] = icon_str
        print(f"天氣資訊更新完成: 平均溫度 {avg_temp//2}°C，圖示 {icon_str}")
    except aiohttp.ClientError as e:
        print(f"請求錯誤: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"資料解析錯誤: {e}")

async def response_synthesizer(data: str, conn: socket.socket, LLM_handler: ChatGPTResponseHandler, emotion_analyzer: StreamEmotionAnalyzer) -> None:
    """生成回應"""
    sentence_buffer = ""
    full_response = []
    processed_sentences = set()

    async for text in LLM_handler.generate_response(data):
        sentence_buffer += text
        sentences = re.split(r'(?<=[。！？])', sentence_buffer)

        for sent in sentences[:-1]:
            clean = sent.strip().replace(" ", "")
            if not clean or clean in processed_sentences:
                continue

            processed_sentences.add(clean)  # 標記已處理過
            print(f"\nAI回傳語句: {clean}")
            full_response.append(clean)

            # 情緒辨識
            emotion_analyzer.feed(clean)
            emotion = emotion_analyzer.get_emotion()

            # 進行語音合成
            send_tts(clean, emotion, conn)
        # 如果最後一段沒句號，則留著下次繼續累積
        if not sentence_buffer.endswith(("。", "！", "？")):
            sentence_buffer = sentences[-1]
        else:
            sentence_buffer = ""
    
    full_response_str = ''.join(full_response)
    print(f"\nAI回傳完整回應: {full_response_str}")
    # 在每段文字回復之後更新聊天歷史
    LLM_handler.HistoryManager.append_chat_history(
        {"role": "assistant","content": ''.join(full_response)}
        )
    
    # 在更新後儲存聊天歷史到 JSON 檔案
    # TODO: 由於IO處理可能較為耗時，這裡未來會考慮非同步寫入
    # 或是可以決定一個"批量寫入"時機
    LLM_handler.HistoryManager.save_to_json()

async def handle_client(conn: socket.socket, addr: tuple) -> None:
    """處理 Unity 請求
    TODO: 注意目前的架構只是「同步處理每個任務」
    需要在架構上做調整才能真正非同步處理多個任務步驟
    """
    print(f"連線來自 {addr}")

    #初始化模型
    LLM_Model = ChatGPTResponseHandler(connection_to_UI=conn)

    #初始化天氣資訊
    await asyncio.create_task(update_weather_info(LLM_Model, conn))

    # 初始化聊天歷史管理器
    LLM_chat_history = ChatHistoryManager("chat_history.json")

    LLM_Model.HistoryManager = LLM_chat_history

    # 若有多個聊天紀錄檔案，建議用一個 HistoryManager 物件對應一組聊天紀錄檔案
    if LLM_chat_history.chat_history:
        print("聊天紀錄已讀取，準備處理請求...")
    else:
        print("沒有找到聊天紀錄，將創建新的空白紀錄。")
    
    # 主迴圈，持續接收 Unity 發來的請求
    # 並不在event loop中執行，因為socket本身是阻塞的
    while True:
        try:
            # 接收來自 Unity 的資料
            data = conn.recv(1024).decode("utf-8")

            if not data:
                print("未接收到資料或連線已中斷")
                break

            print(f"收到 Unity 訊息: {data}")

            if data == "stt":
                print("開始進行語音辨識...")

                # **語音辨識**
                try:
                    text, voice_emotion = recognize_speech()
                    print(f"語音辨識結果: {text}")
                    emotion_analyzer = StreamEmotionAnalyzer()
                    emotion_analyzer.feed(text)
                    useremotion = emotion_analyzer.get_emotion()
                    print(f"辨識到的用戶情緒: {useremotion}")

                except Exception as e:
                    print(f"語音辨識錯誤: {e}")
                    text = "無法識別語音"
                
                data = "用戶文字情緒:"+useremotion+", 語音情緒:"+voice_emotion+"\n"+text
                conn.sendall(f"\nUser_Emotion:{voice_emotion}\n".encode("utf-8"))

            if not data:
                print("接收到空資料，跳過處理")
                continue
            print(f"🔹 傳送至 LLM 處理: {data}")

            # 生成回應
            await asyncio.create_task(response_synthesizer(
                data=data.strip(),
                conn=conn,
                LLM_handler=LLM_Model,
                emotion_analyzer=StreamEmotionAnalyzer()
                ))
            print(f"針對{data} 的回應已生成。")
        except :break

    conn.close()
    raise SystemExit("連線結束")

# 啟動主伺服器
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind((HOST, PORT))
    server.listen()
    print("主控伺服器啟動，等待 Unity 連線...")

    while True:
        conn, addr = server.accept()
        try:
            asyncio.run(handle_client(conn, addr))
        except SystemExit as exit_msg:
            print(exit_msg)
            conn.close()
            break
        except Exception as e:
            print(f"處理客戶端時發生錯誤: {e}")
            conn.close()
            break
