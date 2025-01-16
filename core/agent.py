from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent
from core.llm import llm
from core.tools import tools

# Memory
memory = ConversationBufferMemory(return_messages=True)

# Agent
agent_with_memory = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    system_prompt = """
    你是一個智能助理，並以繁體中文回答問題，可以回答各種類型的問題。
    """
)
