from langchain_community.chat_models import ChatOpenAI

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import OPENAI_BASE_URL, OPENAI_KEY

llm = ChatOpenAI(model="gpt-4o",temperature=0,base_url=OPENAI_BASE_URL,api_key=OPENAI_KEY)

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("introduce yourself"),
]

answer = llm.invoke(messages)

print(answer)