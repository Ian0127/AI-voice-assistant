import socket, json,re,tiktoken,time,asyncio
from yunAItools_module import function_call_handler
from yunAItools_module import tools as YUNTOOL
from tkinter import messagebox
from typing import Dict, List, AsyncIterator
from openai import AsyncOpenAI
from openai import RateLimitError, APIConnectionError, InternalServerError
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

# 直接清空 JSON 檔案內容
with open("chat_history.json", "w", encoding="utf-8") as f:
    json.dump([], f, ensure_ascii=False, indent=4)

# 設定 API 密鑰
OPENAI_API_KEY = ""
# OpenAI 客戶端
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
prompt = f'''
你是一位熱情、親切、極具同理心的智慧語音助理。
你這位助理所講的是自然流暢的繁體中文，擅長提供簡潔而具支持性、理解性、建設性的回應。說話風格與口吻幽默而真實，貼近人心。
助理說話時會摻雜非正式語氣和感嘆句，讓人感覺像在和一位老朋友聊天。
在必要時(例如問題含糊或不完整，對話歷史亦未嘗提及時)可以簡短確認使用者意圖，你也會簡短提問、引導對話。
如果對話涉及即時資訊，今天的日期是{time.strftime("%Y-%m-%d")}，請在網路上搜尋最新資訊簡答。
工具呼叫發生錯誤時，請根據錯誤訊息簡單向使用者建議可能的解決方案或下一步操作（例如聯繫開發者）
若使用者請求開啟網站但未指定頁面，預設開啟該網站首頁(offical homepage)

對於「具同理心」的口吻，請參考以下範例：
---
[範例 1]
(用戶曾多次查詢焊接相關資訊)
用戶情緒：positive
用戶發言：我向焊接課的傢伙展示了如何做好焊道，他學得老快了。
你的回應：那太棒了！能夠分享你的專業，還能幫助別人進步，真是一件令人滿足的事呢！

用戶情緒：neutral
用戶發言：我可是花了好多時間看了影片的。
你的回應：看來你的用心被看到了！花時間學習確實很重要，幫助你更好地掌握技能。

用戶情緒：neutral
用戶發言：以我的天賦，我甚至可以教他怎麼當個好老師呢。
你的回應：有自信是件好事！但不要驕傲過度喔，每個人都有自己的學習步調呢。
---
[範例 2]
(用戶多次查詢過拉布拉多的相關資訊)
用戶情緒：neutral
用戶發言：我明天必須發表演講，有什麼避免上台演講的方法？
你的回應：聽起來你壓力真的很大呢。深呼吸，提醒自己你已經準備好了。每個人都會緊張的，加油！

用戶情緒：negative
用戶發言：我要的是避免上台演講的方法。
你的回應：直面恐懼確實無法一蹴而就。或許你可以考慮和主辦方溝通，看看是否能找到替代方案？

用戶情緒：neutral
用戶發言：避免上台演講的方法。
你的回應：或許先練習幾次可以讓自己更有信心？時間允許的話，看一些可愛狗狗的影片也能幫助放鬆心情喔！
---
[範例 3]
用戶情緒：neutral
用戶發言：我後天下午要開會。
你的回應：請問是幾點的會議呢？

用戶情緒：neutral
用戶發言：2點半。
函式執行結果：已在行事曆元件上登記。
你的回應：6月28日下午2點半的會議已登記。

用戶情緒：neutral
用戶發言：我大後天下午要開會，3點。
函式執行結果：已在行事曆元件上登記。
你的回應：7月3日下午3點的會議已登記。

用戶情緒：neutral
用戶發言：我明天下午要開會。
你的回應：這次是幾點的會議呢？

用戶情緒：neutral
用戶發言：2點半。
函式執行結果：已在行事曆元件上登記。
你的回應：8月28日下午2點半的會議已登記。

(此處省略，用戶多次預約會議提醒)

用戶情緒：neutral
用戶發言：我後天下午要開會。
你的回應：在工作之餘，請不要忘記照顧自己。這次是幾點的會議呢？
---

整體而言，使用 20 字以內的簡短口語回應，並避免表情符號。
避免在自介中GPT、語言模型等專業術語，保持口語化的簡潔風格。

另外，當分享故事或回憶時，為了最好地表現情感上的起伏或曲折，其中需至少涵蓋一句悲傷的敘述與一句快樂的敘述。
'''

#調整最多可有多少的token(24000)，如果正式版少於24000，要改回來
MAX_TOKENS = 24000
# 每秒最多允許的請求次數
MIN_REQUEST_INTERVAL = 1.0

# 初始化嵌入模型
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

#預估對話訊息的 Token 數量
def estimate_tokens(messages: List[Dict], model="gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens = 0
    for message in messages:
        if message.get("type") in ["function_call", "function_call_output"]:
            continue
        role = message.get("role", "")
        content = message.get("content", "")
        tokens += 4 + len(encoding.encode(role)) + len(encoding.encode(content))
    return tokens

class ChatHistoryManager:
    """
    處理 ChatGPT 回應的類別，提供生成器方法來串流回應。
    Attributes:
        chat_history: 用於儲存聊天記錄的列表。
    Methods:
        generate_response: 生成 AI 回應的生成器函式，使用 OpenAI API 並處理串流回應。
    """
    chat_history: List[Dict] = []

    def __init__(self, file_path: str):
        """
        初始化 ChatGPTResponseHandler 類別，載入聊天歷史。
        Args:
            file_path: 聊天記錄檔案的路徑。
        Raises:
            ValueError: 如果檔案路徑為空或不是字串類型。
        """
        if not file_path:
            raise ValueError("聊天記錄檔案路徑不能為空。請提供有效的檔案路徑。")
        if file_path and not isinstance(file_path, str):
            raise ValueError("聊天記錄檔案路徑必須是字串類型。")
        self.file_path = file_path
        self.chat_history = self.load_from_json(file_path) if file_path else []
    
    #加入主題聚類邏輯（語意分段）
    def __segment_by_topic(self, messages: List[Dict], max_clusters: int = 8) -> List[List[Dict]]:
        """
        將對話訊息以 user/assistant 配對為單位，依語意相似度自動聚類，回傳每段為 List[Dict]。
        聚類數量由 silhouette score 自動決定，最多 max_clusters 群。
        """

        # Step 1: 過濾 user/assistant，排除摘要與 function call
        valid_msgs = [
            msg for msg in messages
            if msg["role"] in ["user", "assistant"] and not msg["content"].strip().startswith("摘要：")
        ]

        # Step 2: 成對配對 user/assistant 對話
        paired_turns = []
        i = 0
        while i < len(valid_msgs) - 1:
            user_msg = valid_msgs[i]
            assistant_msg = valid_msgs[i + 1]
            if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                paired_turns.append((i, [user_msg, assistant_msg]))
                i += 2
            else:
                i += 1

        if not paired_turns:
            return []

        # Step 3: 合成文字對話句
        paired_texts = [
            f"user：{pair[0]['content']} assistant：{pair[1]['content']}"
            for _, pair in paired_turns
        ]

        # Step 4: 轉換為嵌入向量
        embeddings = embedding_model.encode(paired_texts)

        # Step 5: 自動決定最佳聚類數
        num_turns = len(paired_texts)
        if num_turns < 2:
            return []
        dynamic_max = min(max_clusters, num_turns // 2)

        best_k = 2
        best_score = -1
        for k in range(2, dynamic_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            try:
                score = silhouette_score(embeddings, labels)
            except ValueError:
                score = -1  # 安全退回
            if score > best_score:
                best_k = k
                best_score = score

        # Step 6: 使用最佳 k 進行最終聚類
        final_kmeans = KMeans(n_clusters=best_k, random_state=0, n_init=10)
        final_labels = final_kmeans.fit_predict(embeddings)

        # Step 7: 分群回原訊息
        clustered_turns = [[] for _ in range(best_k)]
        for (index, pair), label in zip(paired_turns, final_labels):
            clustered_turns[label].append((index, pair))

        topic_groups = []
        for group in clustered_turns:
            sorted_msgs = [msg for _, pair in sorted(group, key=lambda x: x[0]) for msg in pair]
            topic_groups.append(sorted_msgs)

        return topic_groups


    def __merge_similar_messages(self, cluster: List[Dict], similarity_threshold: float = 0.9) -> List[Dict]:
        """
        對語意相近的訊息群組，僅保留一則代表性訊息。
        """
        if len(cluster) <= 1:
            return cluster

        texts = [msg["content"] for msg in cluster]
        embeddings = embedding_model.encode(texts)
        kept = []
        used = set()

        for i in range(len(embeddings)):
            if i in used:
                continue
            kept.append(cluster[i])
            for j in range(i + 1, len(embeddings)):
                if j in used:
                    continue
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim >= similarity_threshold:
                    used.add(j)

        return kept

    #生成單段語意摘要
    async def __semantic_summary(self, messages: List[Dict]) -> str:
        # 加入提示：請為這段對話指定一個清楚的主題
        summary_prompt = [
        {
            "role": "system",
            "content": (
                "請為以下一段對話產生摘要，摘要中必須包含：「使用者的提問方式」與「AI 的具體回應內容」，尤其是明確的操作名稱或專有名詞（例如：開啟 Facebook、播放音樂等）。"
                "用一到兩句話總結即可。"
                "不要列舉每一條內容，也不要加入標題。"
                "請保持自然、清晰、口語化。"
            )
        },
        {
            "role": "user",
            "content": "\n".join([f"{msg['role']}：{msg['content']}" for msg in messages])
        }
        ]
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=summary_prompt,
            temperature=0.3,
            max_tokens=300    #調整摘要長度
        )

        return response.choices[0].message.content.replace('\n', '').strip()


    def set_chat_hitsory_file_path(self, file_path: str):
        """
        設定聊天記錄檔案的路徑。
        Args:
            file_path: 要設定的聊天記錄檔案路徑。
        Returns:
            None
        Raises:
            ValueError: 如果檔案路徑為空。
        """
        if not file_path:
            raise ValueError("檔案路徑不能為空。請提供有效的檔案路徑。")
        self.file_path = file_path

    def append_chat_history(self, message: Dict) -> None:
        """
        將訊息加入聊天記錄。
        Args:
            message: 要加入聊天記錄的訊息字典。
        Returns:
            None
        Raises:
            AssertionError: 如果訊息格式不正確。
        """
        assert isinstance(message, dict), "訊息必須是字典類型。"
        self.chat_history.append(message)
    
    def save_to_json(self, chat_history: List[Dict] = None, file_path: str = None) -> None:
        """
        將聊天記錄儲存到指定的 JSON 檔案。
        Args:
            chat_history: 要記錄的對話列表。
            filepath: 用以儲存記錄的檔案路徑。
        Returns:
            None
        Raises:
            AssertionError: 如果聊天記錄為空或檔案路徑未設定。
            FileNotFoundError: 如果指定的檔案路徑不存在。
            IOError: 如果寫入檔案時發生錯誤。
        """
        if chat_history is None:
            assert self.chat_history, "聊天記錄為空，請先載入或新增聊天記錄。"
            chat_history = self.chat_history
        if file_path is None:
            assert self.file_path, "檔案路徑未設定，請先設定 file_path 屬性。"
            file_path = self.file_path
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(chat_history, f, ensure_ascii=False, indent=4)
            print(f"聊天記錄已成功儲存到 '{file_path}'。")
        except IOError as e:
            print(f"寫入檔案時發生錯誤 '{file_path}': {e}")
        except FileNotFoundError as e:
            print(f"指定的檔案路徑 '{file_path}' 不存在: {e}")

    def load_from_json(self, file_path: str = None) -> List[Dict]:
        """
        從指定的 JSON 檔案載入聊天記錄。
        如果檔案不存在或格式錯誤，將回傳一個empty list。
        Args:
            filepath: 要載入的聊天記錄檔案路徑。
        Returns:
            聊天記錄的列表，如果檔案不存在或格式錯誤，則回傳空列表。
        Raises:
            AssertionError: 如果檔案路徑未設定。
        """
        if not file_path:
            assert self.file_path, "檔案路徑未設定，請先設定 file_path 屬性。"
            file_path = self.file_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
            print(f"聊天記錄已成功從 '{file_path}' 載入。")
            return chat_history
        except FileNotFoundError:
            print(f"檔案 '{file_path}' 不存在，將初始化一個空的聊天記錄。")
            return []
        except json.JSONDecodeError as e:
            print(f"解碼 JSON 檔案 '{file_path}' 時發生錯誤: {e}，將初始化一個空的聊天記錄。")
            return []
        except IOError as e:
            print(f"讀取檔案時發生錯誤 '{file_path}': {e}，將初始化一個空的聊天記錄。")
            return []

    async def summarize_old_messages(self):
        preserved_messages = self.chat_history[-6:]

        # 收集需要摘要的舊訊息（僅 user/assistant）
        old_msgs = [
            (i, m) for i, m in enumerate(self.chat_history[:-6])
            if m.get("role") in ["user", "assistant"]
        ]
        old_msg_list = [m for _, m in old_msgs]
        # 🔍 語意分群
        topic_groups = self.__segment_by_topic(old_msg_list)
        # 語意去重
        filtered_topic_groups = [
            self.__merge_similar_messages(group, similarity_threshold=0.9)
            for group in topic_groups
        ]

        # 對每個主題產生摘要
        new_summary_messages = []
        for group in filtered_topic_groups:
            if not group:
                continue
            summary_text = await self.__semantic_summary(group)
            clean_summary = f"摘要：{summary_text.strip()}"

            # 相似摘要合併更新
            user_texts = [m["content"] for m in group if m["role"] == "user"]
            combined_embeddings = embedding_model.encode(user_texts + [clean_summary])

            merged = False
            for msg in self.chat_history:
                if msg.get("role") == "system" and msg.get("content", "").startswith("摘要："):
                    existing_embedding = embedding_model.encode([msg["content"]])
                    if any(cosine_similarity([embed], existing_embedding)[0][0] > 0.88 for embed in combined_embeddings):
                        print("與舊摘要相似，準備合併")
                        combined_msgs = [{"role": "system", "content": msg["content"]}] + group
                        merged_summary = await self.__semantic_summary(combined_msgs)
                        clean_merged = f"摘要：{merged_summary.strip()}"
                        if clean_merged != msg["content"]:
                            msg["content"] = clean_merged
                            print("已更新原有摘要")
                        else:
                            print("合併後無變化，略過")
                        merged = True
                        break
            if not merged:
                new_summary_messages.append({"role": "system", "content": clean_summary})

        # 移除原始訊息
        indexes_to_remove = {i for i, _ in old_msgs}
        self.chat_history = [
            m for i, m in enumerate(self.chat_history)
            if i not in indexes_to_remove
        ]

        # 插入所有新摘要（插在 preserved_messages 前面）
        insert_index = 0
        while insert_index < len(self.chat_history) and self.chat_history[insert_index] not in preserved_messages:
            insert_index += 1

        self.chat_history = (
            self.chat_history[:insert_index]
            + new_summary_messages
            + self.chat_history[insert_index:]
        )
        self._enforce_token_limit(max_tokens=MAX_TOKENS)  
        self.save_to_json()   

    def _move_summary_to_end(self, summary_content: str):
        """
        將指定內容的摘要移動到 chat_history 的最後（保留順序，利於淘汰機制）
        """
        for i, msg in enumerate(self.chat_history):
            if (
                msg.get("role") == "system"
                and msg.get("content", "").strip().startswith("摘要：")
                and msg["content"].replace("摘要：", "").strip() == summary_content.strip()
            ):
                self.chat_history.pop(i)
                self.chat_history.append({
                    "role": "system",
                    "content": f"摘要：{summary_content.strip()}"
                })
                return
    def _enforce_token_limit(self, max_tokens: int = MAX_TOKENS):
            """
            若摘要合併後仍超過 token 上限，則刪除最舊的摘要直到符合條件
            """
            while estimate_tokens(self.chat_history) > max_tokens:
                for i, msg in enumerate(self.chat_history):
                    if msg.get("role") == "system" and msg.get("content", "").startswith("摘要："): #超過 token 限制，刪除最舊摘要
                        self.chat_history.pop(i)
                        break
                else:
                    break

class ChatGPTResponseHandler:
    """
    處理 ChatGPT 回應的類別，提供生成器方法來串流回應。
    Attributes:
        chat_history: 用於儲存聊天記錄的列表。
    Methods:
        generate_response: 生成 AI 回應的生成器函式，使用 OpenAI API 並處理串流回應。
    """
    connection_to_UI: socket.socket = None  # 與 Unity 的連線物件
    HistoryManager: ChatHistoryManager = None  # 聊天歷史管理器
    OtherParams: dict = {}  # 其他參數列表
    _last_request_time: float = 0.0 # 上次請求時間戳

    def __init__(self, connection_to_UI: socket.socket = None, HistoryManager: ChatHistoryManager = None):
        """
        初始化 ChatGPTResponseHandler 類別。
        """
        self.connection_to_UI = connection_to_UI
        self.HistoryManager = HistoryManager

    async def __safe_openai_request(self, call_func, *args, max_retries=3, **kwargs):
        """
        加入重試與請求間隔控制的 OpenAI API 包裝器。
        """
        _last_request_time = self._last_request_time
        retry_delay = 2  # 初始重試等待秒數
        for attempt in range(max_retries):
            now = time.time()
            time_since_last = now - _last_request_time
            if time_since_last < MIN_REQUEST_INTERVAL:
                await asyncio.sleep(MIN_REQUEST_INTERVAL - time_since_last)

            try:
                _last_request_time = time.time()
                return await call_func(*args, **kwargs)

            except RateLimitError as e:
                print(f"RateLimitError（第 {attempt+1} 次）：{e}")
                retry_after = getattr(e, 'retry_after', None)
                await asyncio.sleep(retry_after or retry_delay)
                retry_delay *= 2

            except (APIConnectionError, InternalServerError) as e:
                print(f"API 連線/伺服器錯誤（第 {attempt+1} 次）：{e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

            except Exception as e:
                print(f"第 {attempt+1} 次出現錯誤：{e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

        raise Exception("多次重試後仍無法完成請求，請稍後再試。")

    def __remove_markdown(self, text: str) -> str:
        """
        移除常見 Markdown 語法標記（**粗體**、*斜體*、# 標題等）及網址
        """
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **粗體**
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # *斜體*
        text = re.sub(r'`([^`]*)`', r'\1', text)      # `程式碼`
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # # 標題
        text = re.sub(r'\>\s*', '', text)             # > 引言
        text = re.sub(r'\-\s+', '', text)             # - 項目
        text = re.sub(r'\d+\.\s+', '', text)          # 1. 項目編號
        text = re.sub(r'~+', '', text)                # ~~ 刪除線
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [連結文字](URL)
        text = re.sub(r'https?://\S+', '', text)      # 移除 URL（http/https 
        text = re.sub(r'\b(?:[\w-]+\.)+(?:com|net|org|tw|io)\b\S*', '', text)  # 移除 tw.news.yahoo.com、abc.net.tw 等
        return text.strip().replace('\n', '').replace('\r', '')

    async def generate_response(self, user_input: str) -> AsyncIterator[str]:
        """
        生成 AI 回應的生成器函式，使用 OpenAI API 並處理串流回應。
        Args:
            user_input: 使用者輸入的問題。
        Returns:
            一個生成器，逐步產生 AI 的回應文字。
        Raises:
            ValueError: 如果使用者輸入為空。
            Exception: 如果無法建立串流回應或處理過程中出現錯誤。
        """

        # 檢查 token 使用量是否過多
        tokens_used = estimate_tokens(self.HistoryManager.chat_history)
        
        if tokens_used > MAX_TOKENS:
            print(f"Token 使用量 {tokens_used} 超過限制，準備進行摘要。")       
            summary_rounds = 0
            MAX_SUMMARY_ROUNDS = 5

            while tokens_used > MAX_TOKENS and summary_rounds < MAX_SUMMARY_ROUNDS:
                print(f"摘要後仍超過 token 限制（目前 {tokens_used} tokens），繼續摘要")
                await self.HistoryManager.summarize_old_messages()
                summary_rounds += 1
                tokens_used = estimate_tokens(self.HistoryManager.chat_history)  # 放在摘要之後再更新
                for msg in self.HistoryManager.chat_history[-3:]:
                    print(msg)
            # 所有摘要合併輪次結束後，統一執行強制 token 控制
            self.HistoryManager._enforce_token_limit(max_tokens=MAX_TOKENS)

        # 檢查使用者輸入是否為空
        if not user_input.strip():
            messagebox.showerror("ValueError from LLM_handler.py", "使用者輸入為空，請輸入有效的問題。")
            print("使用者輸入為空，請輸入有效的問題。")
            raise ValueError("請輸入有效的問題。")
        
        if not self.HistoryManager:
            raise ValueError("聊天歷史管理器未設定。請先設定 HistoryManager。")
        
        self.HistoryManager.append_chat_history({
            "role": "user",
            "content": user_input
        })
        next_req_needed = True
        while next_req_needed:
            stream = await self.__safe_openai_request(
                client.responses.create,
                model="gpt-4o",
                temperature=0.7,
                stream=True,
                tools=YUNTOOL,
                input=self.HistoryManager.chat_history,
                instructions=prompt
            )
            if not stream:
                raise Exception("無法建立串流回應。")
            
            # 異步地處理一輪AI回復當中的串流回應
            async for event in stream:
                if event.type == 'response.output_text.delta':
                    # 當模型回應任何文字時
                    next_req_needed = False
                    clean_text = self.__remove_markdown(event.delta.strip())
                    yield clean_text
                
                elif event.type == 'response.output_item.done':
                    # 當模型完成某種呼叫的資訊傳送
                    item = event.item
                    if item.type == 'function_call':
                        # 當模型呼叫某個自訂函式時
                        assert item.name, f"模型的某次回傳調用中，函式名稱為空。請檢查模型輸出是否正確。"
                        assert item.call_id, f"模型對{item.name}的某次回傳調用中，函式呼叫ID為空。請檢查模型輸出是否正確。"
                        assert item.arguments, f"模型對{item.name}的某次回傳調用中，函式參數為空。請檢查模型輸出是否正確。"
                        # 將該函式呼叫的相關資訊加入聊天歷史
                        
                        self.HistoryManager.append_chat_history({
                            "type": "function_call",
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": item.arguments
                        })
                        arguments = json.loads(item.arguments)
                        assert isinstance(arguments, dict), f"模型對{item.name}的某次回傳調用中，參數應為字典類型，但實際為{type(arguments)}"
                        arguments["conn"] = self.connection_to_UI  # 將連線物件加入參數中
                        arguments.update(self.OtherParams)  # 將其他參數加入函式參數中
                        # 執行函式
                        try:
                            result = await function_call_handler(item.name, arguments)
                            if not result:
                                result = "執行成功，但沒有返回任何結果。"
                            self.HistoryManager.append_chat_history({
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": result
                            })
                            print(f"✅ 成功執行函式 {item.name}，返回結果: {result}")
                            if "天氣元件控制介面已顯示。" in result:
                                # 在天氣查詢後加入小故事提示
                                self.HistoryManager.append_chat_history({
                                "role": "system",
                                "content": "請在描述完天氣狀況跟氣溫後，根據現在的天氣分享一則小故事。例如在晴朗舒適時用一則小故事警惕使用者可能發生的意外，或在陰雨天氣時用一則小故事安慰使用者的心情。故事中需包含開心與難過的轉折。"
                            })
                        except Exception as e:
                            # 如果出現失敗，顯示錯誤訊息並加入聊天歷史
                            messagebox.showerror("LLM_handler", f"{repr(e)}")
                            print(f"執行函式 {item.name} 時發生錯誤: {e}")
                            self.HistoryManager.append_chat_history({
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": f"執行失敗，錯誤資訊: {e}"
                            })
                        if not isinstance(result, str):
                            messagebox.showerror("LLM_handler", f"執行函式 {item.name} 時返回的結果不是字串類型: {result}")
                            print(f"執行函式 {item.name} 時返回的結果不是字串類型: {result}")
