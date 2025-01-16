from langchain_core.vectorstores import InMemoryVectorStore
from core.llm import embedding_model

vectorstore = InMemoryVectorStore(embedding=embedding_model)
