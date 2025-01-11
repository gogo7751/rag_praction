import os
from langchain.llms import AzureOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 從環境變數取得 Azure OpenAI 的設定
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# 設定 Azure OpenAI LLM
llm = AzureOpenAI(
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=AZURE_OPENAI_API_KEY,
    openai_api_base=AZURE_OPENAI_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_VERSION,  # 確認使用的 API 版本
)

# 使用記憶功能來建立聊天鏈
memory = ConversationBufferMemory()
conversation_chain = ConversationChain(llm=llm, memory=memory)


# 測試聊天功能
def main():
    print("Azure OpenAI 與 LangChain 聊天機器人 (輸入 'exit' 結束)")
    while True:
        user_input = input("你: ")
        if user_input.lower() == "exit":
            print("再見！")
            break

        response = conversation_chain.run(user_input)
        print(f"AI: {response}")


if __name__ == "__main__":
    main()
