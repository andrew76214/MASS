from langchain_openai import ChatOpenAI
from langchain_ollama import Chatollama

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

# Supervisior
## Self-RAG
## Writer
## Note taker

summary_group_llm = ChatOpenAI(model="gpt-4o")
