import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
import time
from Operator import BaseOperator
from utils import extract_module_code

class Retrieval(BaseOperator):
    """
    Retrieval Operator that retrieves relevant information from a knowledge base
    """
    def __init__(self,model: str, temperature: float=0.5,  topK:int=3,**kwargs):
        super().__init__(model, temperature, **kwargs)
        if model != "bge-m3":
            raise ValueError("Retrieval Operator only supports bge-m3 model.")
        if topK <= 0:
            raise ValueError("topK must be a positive integer.")
        # self.embedding_model = self.get_llm(model)
        self.topK = topK
        self.mission_type = "RIOT"  # Default mission type, can be changed as needed
        try:
            self.client = self.get_llm(model)
            self.collection = self.client.get_collection(
                name=self.mission_type+"_embedding",
                embedding_function=ImprovedEmbeddingFunction(model_name=model)
                )
        except Exception as e:
            raise ValueError(f"Failed to connect to the ChromaDB collection: {e}")
    def _run(self, query=None):
        """
        RAG
        """
        response =  self.collection.query(
            query_texts=query,
            n_results=self.topK,
        )

        return str(response)

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
            # 返回零向量作为fallback
            return [[0.0] * 1024 for _ in input]  # 假设向量维度为1024   


if __name__ == "__main__":
    node = Retrieval(model="bge-m3", mission_type="RIOT", topK=3)
    response = node.run(query="""
    I need to develop RIOT code on an ESP32, which sends a CoAP request to an COAP server. The program is configured to send CoAP POST requests "Hello, COAP Cloud" to the COAP server at IP address "47.102.103.1" and port 5683, with the URI path "/coap/test".
""")
    print(response)