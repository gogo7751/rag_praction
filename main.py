from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import os

# 載入環境變數
load_dotenv()

# Azure OpenAI 配置
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

AZURE_OPENAI_API_EMBEDDING_KEY = os.getenv("AZURE_OPENAI_API_EMBEDDING_KEY")
AZURE_OPENAI_API_EMBEDDING_RESOURCE_NAME = os.getenv("AZURE_OPENAI_API_EMBEDDING_RESOURCE_NAME")
AZURE_OPENAI_API_EMBEDDING_VERSION = os.getenv("AZURE_OPENAI_API_EMBEDDING_VERSION")
AZURE_OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_API_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_API_EMBEDDING_MODEL")

# 定義消息格式
class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

# 設定 Azure Chat OpenAI
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION,
)

# 使用 Azure OpenAI 嵌入模型
embedding_model = AzureOpenAIEmbeddings(
    model=AZURE_OPENAI_API_EMBEDDING_MODEL,
    azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_EMBEDDING_VERSION,
    openai_api_key=AZURE_OPENAI_API_EMBEDDING_KEY,
)

# 初始化向量資料庫
vectorstore = InMemoryVectorStore(embedding=embedding_model)

# 可選：使用本地嵌入模型（如需要可切換）
"""
class SentenceTransformerWrapper:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512

    def embed_documents(self, texts):
        #嵌入多個文檔
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        #嵌入單個查詢
        return self.model.encode([text], convert_to_numpy=True)[0]

# 初始化本地嵌入模型
embedding_model = SentenceTransformerWrapper("all-MiniLM-L6-v2")
"""

# OCR 功能：提取圖片中的文字
def extract_text_from_image(image_path: str) -> str:
    try:
        print(f"正在提取圖片中的文字：{image_path}")
        text = pytesseract.image_to_string(Image.open(image_path), lang="chi_tra+eng")
        print(f"提取的文字：\n{text}")
        return text.strip()
    except Exception as e:
        print(f"OCR 提取文字時出錯: {e}")
        return ""

# 透過 LLM 處理表格數據
def process_with_llm(text: str) -> str:
    try:
        instruction = (
            "請將以下文本轉換為 JSON 格式，按表格欄位結構化：\n\n"
            f"{text}"
        )
        response = llm.generate([[HumanMessage(content=instruction)]])
        structured_data = response.generations[0][0].text.strip()
        print("LLM 整理後的數據：")
        print(structured_data)
        return structured_data
    except Exception as e:
        print(f"透過 LLM 處理表格數據時出錯: {e}")
        return ""

# 將處理後的數據存入向量資料庫
def process_text_to_vector(text: str, metadata: dict):
    try:
        vectorstore.add_texts(
            texts=[text],
            metadatas=[metadata],
        )
        print("文字已成功存入向量資料庫！")
    except Exception as e:
        print(f"存入向量資料庫時出錯: {e}")

# 基於 RAG 流程的用戶問題處理
def handle_user_query(query: str):
    try:
        # 初始化檢索器
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("檢索器初始化成功！")

        # 使用檢索器進行檢索
        results = retriever.get_relevant_documents(query)
        print(f"檢索到 {len(results)} 條相關資料")

        if not results:
            return "資料庫內無相關內容"

        # 提取檢索結果中的文本內容
        context = "\n".join(doc.page_content for doc in results if hasattr(doc, "page_content"))
        if not context.strip():
            return "資料庫內無相關內容"

        # 設定 System Prompt 並生成 LLM 回應
        system_prompt = (
            "請依照檢索的答案回答。如果檢索的內容無法回答問題，請回答「資料庫內無相關內容」。"
        )
        response = llm.generate([
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"以下是檢索的相關內容：\n\n{context}\n\n用戶問題：{query}"),
            ]
        ])
        return response.generations[0][0].text.strip()

    except Exception as e:
        return f"處理用戶問題時出錯: {e}"

# 主函數
def main():
    # 圖片路徑
    image_path = "source/table.png"  # 確保路徑正確
    if not os.path.exists(image_path):
        print(f"圖片 {image_path} 不存在！")
        return

    # OCR 提取圖片內容
    extracted_text = extract_text_from_image(image_path)
    if not extracted_text:
        print("無法從圖片中提取文字。")
        return

    # 使用 LLM 將文字數據結構化
    structured_data = process_with_llm(extracted_text)
    if structured_data:
        # 存入向量資料庫
        metadata = {"source": image_path, "description": "結構化表格數據"}
        process_text_to_vector(structured_data, metadata)
    else:
        print("LLM 未能處理表格數據。")

    print("Azure OpenAI 與 LangChain 聊天機器人 (輸入 'exit' 結束)")

    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("再見！")
            break

        response = handle_user_query(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()
