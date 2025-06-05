import time
from langchain.schema import BaseMessage
from utils import extract_module_code
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from pydantic import SecretStr
from typing import Union
from config import *
import tiktoken
from utils import myllm
import logging
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_ollama import OllamaEmbeddings

class ImprovedEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "bge-m3"):
        self.embed_model = OllamaEmbeddings(model=model_name)
        
    def __call__(self, input: Documents) -> Embeddings:
        try:
            return self.embed_model.embed_documents(input)
        except Exception as e:
            logging.error(f"嵌入计算失败: {str(e)}")
            # 返回零向量作为 fallback
            return [[0.0] * 1024 for _ in input]  # 假设向量维度为1024


class BaseOperator:
    """
    Base class for all operators. 
    """

    def __init__(self, model: str, temperature: float = 0.5, topk = 3, **kwargs):
        self.model = model
        self.temperature = temperature
        self.topk = topk
        self._cost = {"input_token": 0, "output_tokens": 0, "time": 0}
    
    def _run(self, query=None):
        """
        Run the operator on the input or query.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    def _update_cost(self, input_text, response_text):
        """
        Update the cost based on locally calculated token count using tiktoken.
        
        Args:
            input_text (str): The input text sent to the model.
            response_text (str): The output text received from the model.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        input_tokens = len(encoding.encode(str(input_text)))
        output_tokens = len(encoding.encode(str(response_text)))
        
        self._cost["input_token"] += input_tokens
        self._cost["output_tokens"] += output_tokens
    def reset_cost(self):
        """
        Reset the cost to zero.
        """
        self._cost = {"input_token": 0, "output_tokens":0,"time": 0}
        return
    def get_cost(self):
        # return the cost
        return self._cost
    def run(self, query=None):
        """
        Return the response in text format.
        """
        start_time = time.time()
        try:
            response = self._run(query)
        except Exception as e:
            print(f"Error occurred during execution: {e}")
            return None
        end_time = time.time()
        self._cost["time"] += end_time - start_time
        # response = response.content
        # response_format = extract_module_code(response)
        return response.content if isinstance(response, BaseMessage) else response
    
    # def get_llm(self, model: str, temperature: float = 0)->Union[ChatOpenAI,None]:
    #     if model not in CANDIATE_MODEL:
    #         raise ValueError(f"Model {model} is not supported. Supported models are: {CANDIATE_MODEL}")
    #     elif model in ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]:
    #         client = ChatOpenAI(model=model, temperature=temperature, base_url=OPENAI_BASE_URL, api_key=SecretStr(OPENAI_KEY))
    #     elif model in ["bge-m3"]:
    #         # use OllamaEmbeddings 
    #         import chromadb
    #         from config import DB_PATH
    #         client = chromadb.PersistentClient(path=DB_PATH)

    #     return client
    
    def get_llm(self, model: str, temperature: float = 0)->Union[ChatOpenAI,None]:
        if model not in CANDIATE_MODEL:
            raise ValueError(f"Model {model} is not supported. Supported models are: {CANDIATE_MODEL}")
        elif model in ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]:
            client = ChatOpenAI(model=model, temperature=temperature, base_url=OPENAI_BASE_URL, api_key=SecretStr(OPENAI_KEY))
        elif model in ["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:13b"]:
            client = myllm(model=model, temperature=temperature, base_url=OPENAI_BASE_URL, api_key=SecretStr(OPENAI_KEY))
        elif model in ["bge-m3"]:
            # use OllamaEmbeddings 
            import chromadb
            from config import DB_PATH
            client = chromadb.PersistentClient(path=DB_PATH)
        return client
    
    def retrieval_run(self, query: str, topK: int = 3) -> str:
        """
        Standalone function to perform retrieval from ChromaDB.

        Args:
            query (str): Query text.
            model (str): Name of the embedding model (default: "bge-m3").
            mission_type (str): Mission type used to identify collection name (default: "RIOT").
            topK (int): Number of top results to retrieve (default: 3).

        Returns:
            str: JSON-like string containing retrieved documents and metadata.
        """
        model = "bge-m3"
        mission_type = "RIOT"
        if topK <= 0:
            raise ValueError("topK must be a positive integer.")

        try:
            from chromadb import Client as ChromaClient
            client = self.get_llm(model)
            collection = client.get_collection(
                    name=mission_type+"_embedding",
                    embedding_function=ImprovedEmbeddingFunction(model_name=model)
                    )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to ChromaDB collection: {e}")

        try:
            response = collection.query(
                query_texts=[query],
                n_results=topK
            )
            return str(response)
        except Exception as e:
            raise RuntimeError(f"Query failed: {e}")