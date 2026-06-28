import os, asyncio, socket, time
from pyppeteer import launch

async def open_browser_page(url: str) -> None:
    """用瀏覽器開啟網址"""
    # TODO: 新增更多瀏覽器的支援
    # TODO: 偵測使用者的瀏覽器
    # TODO: 讓使用者可以自訂瀏覽器的路徑

    if not url or not isinstance(url, str):
        raise ValueError("請提供有效的網址。")

    # 偵測作業系統類型
    OS_TYPE = os.name  # 'nt' for Windows, 'posix' for Linux/macOS

    # 預設瀏覽器路徑（可能需要調查更可能的實際安裝位置並調整）
    DEFAULT_BROWSER_PATH = {
        "edge": os.path.expanduser("~/../../Program Files (x86)/Microsoft/Edge/Application/msedge.exe") if OS_TYPE == "posix" else os.path.expanduser("~/../../Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        "chrome": "C:/Program Files/Google/Chrome/Application/chrome.exe" if OS_TYPE == "nt" else os.path.expanduser("~/../../Program Files/Google/Chrome/Application/chrome.exe")
    }
    
    # 使用第一個在路徑列表中找到的瀏覽器開啟頁面
    for browser in DEFAULT_BROWSER_PATH:
        if os.path.exists(DEFAULT_BROWSER_PATH[browser]):
            browser_path = DEFAULT_BROWSER_PATH[browser]
            # 啟動瀏覽器
            browser = await launch(
                headless=False,
                executablePath=browser_path,
                autoClose=False,
                handleSIGINT = False,
                handleSIGTERM = False,
                handleSIGUP = False
                )
            break
    else:
        # 如果沒有找到任何瀏覽器，則嘗試pyppeteer預設的瀏覽器
        browser = await launch(
            headless=False,
            autoClose=False,
            handleSIGINT = False,
            handleSIGTERM = False,
            handleSIGUP = False
            )
    # 抓取瀏覽器頁面以便進行跳轉
    pages = await browser.pages()
    if pages:
        page = pages[0]
    else:
        # 若沒有頁面，則先等待瀏覽器啟動以便新建頁面
        await asyncio.sleep(3)
        page = await browser.newPage()
    # 執行跳轉
    await page.goto(url)

# 給模型的工具定義
tools = [
    { # 讓AI能執行網路搜尋，與自訂工具無關。
        "type": "web_search_preview",
        "user_location": {
            "type": "approximate",
            "country": "TW",
            "city": "Yunlin",
            "region": "Douliu",
            "timezone": "Asia/Taipei",
        }
    },
    
    { # 開啟瀏覽器頁面
        "type": "function",
        "name": "open_browser_page",
        "strict": True,
        "description": "Open a web page in the browser with the given URL. ",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to open in the browser. For example, 'https://www.example.com'.",
                }
            },
            "required": ["url"],
            "additionalProperties": False
        }
    },
    { # 時鐘元件控制
        "type": "function",
        "name": "clock_UI_trigger",
        "strict": True,
        "description": "Have the clock gadget pop up.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    },
    { # 天氣元件控制
        "type": "function",
        "name": "weather_UI_trigger",
        "strict": True,
        "description": "Trigger the weather gadget displaying today's info.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
    },
    { # 行事曆元件控制
        "type": "function",
        "name": "calendar_UI_trigger",
        "strict": True,
        "description": "Register the event onto the calendar gadget.",
        "parameters": {
            "type": "object",
            "properties": {
                "timeStamp": {
                    "type": "string",
                    "description": "The timestamp of when the event is to be registered, in the format 'yyyy-MM-dd:HH:mm'. For example, 2023-10-01:14:30 or 2024-05-20:09:00, etc.",
                },
                "eventName": {
                    "type": "string",
                    "description": "The name of the registering event.",
                }
            },
            "required": ["timeStamp", "eventName"],
            "additionalProperties": False
        }
    },
]

async def function_call_handler(function_name: str, args: dict) -> str:
    """處理函式呼叫，根據函式名稱和參數執行相應的操作"""
    # 這裡的 args 是一個字典，包含AI給出的所有參數（在tools裡面設定的參數名稱就是key）
    # conn物件已被手動加入args當中，與AI無關。
    # 回傳值將直接是下一輪回應時AI從這裡「收到」的內容，因此建議設為字串
    if function_name == "open_browser_page":
        url: str = args.get("url")
        await open_browser_page(url)
        return f"瀏覽器已開啟，並跳轉至{url}。"
    elif function_name == "clock_UI_trigger":
        conn: socket.socket = args.get("conn")
        conn.sendall("\nCOMMAND:clock_UI_trigger\n".encode("utf-8"))
        return f"時鐘元件控制介面已顯示。當前時間為: {time.strftime('%H:%M')}，請複述一次時間給使用者聽。記得在午夜或凌晨提醒使用者注意身體，或在早晨提醒使用者小心時間的流逝。"
    elif function_name == "weather_UI_trigger":
        conn: socket.socket = args.get("conn")
        weather: str = args.get("currentWeather", "未知天氣")
        temperature: str = args.get("currentTemp", "未知溫度")
        conn.sendall("\nCOMMAND:weather_UI_trigger\n".encode("utf-8"))
        return f"天氣元件控制介面已顯示。天氣狀況為:{weather}，溫度為: {temperature}°C。"
    elif function_name == "calendar_UI_trigger":
        conn: socket.socket = args.get("conn")
        timeStamp: str = args.get("timeStamp")
        eventName: str = args.get("eventName")
        # TODO: 設定觸發格式(若有)
        print("顯示行事曆元件控制介面")
        conn.sendall(f"\nCOMMAND:schedule_add:{timeStamp}:{eventName}\n".encode("utf-8"))
        return f"已在行事曆元件上登記事件: {eventName}，時間: {timeStamp}。"
    else:
        raise ValueError(f"未知的函式名稱: {function_name}")

# 測試用主程式
if __name__ == "__main__":
    from main_server import handle_client, HOST, PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen()
        print("主控伺服器啟動，等待 Unity 連線...")
        
    while True:
        conn, addr = server.accept()
        try:
            handle_client(conn, addr)
        except SystemExit as exit_msg:
            print(exit_msg)
        except Exception as e:
            print(f"處理客戶端時發生錯誤: {e}")
        finally:
            conn.close()
            print("客戶端連線已關閉")
            break