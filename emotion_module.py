from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
class StreamEmotionAnalyzer:
    def __init__(self, endpoint: str = "https://su-xinyu.cognitiveservices.azure.com/", 
                 key: str = "BTnTgfl5TCB1k44LUaRgGrWo58ZVHnHthFhHH7MMDFVckxkckyaUJQQJ99BDAC3pKaRXJ3w3AAAaACOGF1H6", 
                 translation_endpoint: str = "https://su-xinyu1.cognitiveservices.azure.com/", 
                 translation_key: str = "FwrV1PVeByL5HGWdcBVDz1wTkhxmrFNAGjkTq9xNMUpacfNZG1HnJQQJ99BDAC3pKaRXJ3w3AAAbACOGfx8n"):
        self.client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        self.counter = 0
        self.old_score = 0
        self.text_chunks = []


    def feed(self, chunk: str) -> None:
        """新增文字片段到分析序列中"""
        self.counter +=1
        self.text_chunks.append(chunk)
  
    def get_emotion(self) -> str:
        """對目前所有文字片段進行情緒分析"""
        if not self.text_chunks:
            print("沒東西")
            return "neutral"

        chunks = self.text_chunks

        try:
            responses = self.client.analyze_sentiment(list(chunks))
            weighted_score = 0
            for response in responses:
                if not hasattr(response, 'confidence_scores'):
                    continue
                now_socre = response.confidence_scores.positive - response.confidence_scores.negative
                if self.old_score != 0:
                    weighted_score =  now_socre*0.92 + self.old_score*0.08
                else:
                    weighted_score = now_socre

            if weighted_score == 0:
                return "neutral"
            
            self.old_score = weighted_score
            if self.counter >= 9:
                self.reset()
            if weighted_score > 0.35:
                return "positive"
            elif weighted_score <= 0:
                return "negative"
            else:
                return "neutral"
        except Exception as e:
            print(f"情緒分析錯誤: {e}")
            return "neutral"

    def reset(self) -> None:
        """重置目前分析資料"""
        self.text_chunks.clear()
        self.counter = 0
