from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from typing import Union
from config import *
import re
def get_llm(model: str, temperature: float = 0)->Union[ChatOpenAI,None]:
    if model not in CANDIATE_MODEL:
        raise ValueError(f"Model {model} is not supported. Supported models are: {CANDIATE_MODEL}")
    elif model in ["gpt-4o"]:
        llm = ChatOpenAI(model="gpt-4o", temperature=temperature, base_url=OPENAI_BASE_URL, api_key=SecretStr(OPENAI_KEY))

    return llm

def extract_module_code(text: str) -> list:
    """
    Extract all code blocks wrapped in triple backticks (any language).
    E.g., ```c, ```python, etc.

    Args:
        text (str): The input text containing potential code blocks.

    Returns:
        List[str]: A list of strings representing the extracted code blocks.
    """
    pattern = r"```[a-zA-Z]*\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches