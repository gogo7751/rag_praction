from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from config import Config

llm = AzureChatOpenAI(
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_key=Config.AZURE_OPENAI_API_KEY,
    openai_api_version=Config.AZURE_OPENAI_API_VERSION,
    streaming=True,
)

embedding_model = AzureOpenAIEmbeddings(
    model=Config.AZURE_OPENAI_API_EMBEDDING_MODEL,
    azure_endpoint=Config.AZURE_OPENAI_EMBEDDING_ENDPOINT,
    openai_api_version=Config.AZURE_OPENAI_API_VERSION,
    openai_api_key=Config.AZURE_OPENAI_API_EMBEDDING_KEY,
)


# 可選：使用本地嵌入模型（如需要可切換）
"""
class SentenceTransformerWrapper:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512

    def embed_documents(self, texts):
        # 嵌入多個文檔
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        # 嵌入單個查詢
        return self.model.encode([text], convert_to_numpy=True)[0]

# 初始化本地嵌入模型
embedding_model = SentenceTransformerWrapper("all-MiniLM-L6-v2")
"""