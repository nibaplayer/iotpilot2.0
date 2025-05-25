from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from typing import Union
from config import *
import re
import json
import os
def get_llm(model: str, temperature: float = 0)->Union[ChatOpenAI,None]:
    if model not in CANDIATE_MODEL:
        raise ValueError(f"Model {model} is not supported. Supported models are: {CANDIATE_MODEL}")
    elif model in ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]:
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
    pattern = r"```[a-zA-Z+]*\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def load_workflow_config(workflow_config_file_path: str) -> dict:
    """Load workflow config from json file and return a list of WorkflowConfig namedtuples."""
    if os.path.exists(workflow_config_file_path):
        with open(workflow_config_file_path, "r", encoding="utf-8") as f:
            workflow_config = json.load(f)
        print("Workflow config loaded from file:", workflow_config_file_path)
    else:
        raise FileNotFoundError(f"配置文件 '{workflow_config_file_path}' 未找到！")
    return workflow_config