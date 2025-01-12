from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from pydantic import BaseModel, ValidationError, Field
from typing import List
from dotenv import load_dotenv
import os

# 載入環境變數
load_dotenv()

# Azure OpenAI 配置
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

# 定義消息格式
class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

# 設定 Azure Chat OpenAI
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)

# 測試聊天功能
def main():
    print("Azure OpenAI 與 LangChain 聊天機器人 (輸入 'exit' 結束)")
    messages: List[BaseMessage] = []  # 聊天模型的消息歷史

    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("再見！")
            break

        try:
            # 添加用戶消息到消息歷史
            user_message = HumanMessage(content=user_input)
            messages.append(user_message)

            # 呼叫 AzureChatOpenAI，並傳遞消息歷史
            response = llm.generate([messages])

            # 處理 AI 的回應並添加到消息歷史
            ai_message = AIMessage(content=response.generations[0][0].text)
            messages.append(ai_message)

            print(f"AI: {ai_message.content}")
        except ValidationError as e:
            print(f"消息格式錯誤: {e}")
        except Exception as e:
            print(f"發生錯誤: {e}")
            break

if __name__ == "__main__":
    main()
