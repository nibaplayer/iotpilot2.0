import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_ollama import OllamaEmbeddings  # 统一使用这个导入
import glob
import os
import uuid
import logging
from typing import List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 配置参数
from config import RIOT_ROOT, DB_PATH
COLLECTION_NAME = "RIOT_embedding"
TARGET_DIRS = ["examples", "sys", "tests", "drivers"]
MAX_FILE_SIZE = 1024 * 1024  # 1MB限制
BATCH_SIZE = 50  # 批量处理大小

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

def get_c_files_recursive(root_dir: str, target_dirs: List[str]) -> List[str]:
    """递归获取所有.c文件"""
    all_files = []
    for dir_name in target_dirs:
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.exists(dir_path):
            # 递归搜索所有.c文件
            pattern = os.path.join(dir_path, "**", "*.c")
            files = glob.glob(pattern, recursive=True)
            all_files.extend(files)
            logging.info(f"在目录 {dir_name} 中找到 {len(files)} 个.c文件")
    return all_files

def is_file_processable(file_path: str, max_size: int) -> bool:
    """检查文件是否可以处理"""
    try:
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            logging.warning(f"文件过大，跳过: {file_path} ({file_size} bytes)")
            return False
        if file_size == 0:
            logging.warning(f"空文件，跳过: {file_path}")
            return False
        return True
    except OSError:
        logging.error(f"无法获取文件大小: {file_path}")
        return False

def process_files_in_batches(collection, files: List[str], batch_size: int):
    """批量处理文件"""
    processed_count = 0
    failed_count = 0
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        batch_documents = []
        batch_metadatas = []
        batch_ids = []
        
        for file_path in batch_files:
            try:
                if not is_file_processable(file_path, MAX_FILE_SIZE):
                    continue
                    
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                
                if not text.strip():
                    continue
                
                # 准备批量数据
                batch_documents.append(text)
                batch_metadatas.append({
                    "file_path": file_path,
                    "directory": os.path.dirname(file_path).replace(RIOT_ROOT, "").strip("/"),
                    "file_size": len(text),
                    "file_name": os.path.basename(file_path)
                })
                batch_ids.append(str(uuid.uuid4()))
                
            except Exception as e:
                logging.error(f"处理文件 {file_path} 时出错: {str(e)}")
                failed_count += 1
                continue
        
        # 批量添加到集合
        if batch_documents:
            try:
                collection.add(
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                processed_count += len(batch_documents)
                logging.info(f"批量处理完成: {processed_count}/{len(files)} 个文件")
            except Exception as e:
                logging.error(f"批量添加失败: {str(e)}")
                failed_count += len(batch_documents)
    
    return processed_count, failed_count

def main():
    """主函数"""
    try:
        # 初始化ChromaDB客户端
        client = chromadb.PersistentClient(path=DB_PATH)
        
        # 删除现有集合（如果存在）
        try:
            client.delete_collection(COLLECTION_NAME)
            logging.info(f"已删除现有集合: {COLLECTION_NAME}")
        except ValueError:
            logging.info(f"集合 {COLLECTION_NAME} 不存在，跳过删除")
        
        # 创建新集合
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ImprovedEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"}  # 修正拼写错误
        )
        logging.info(f"已创建新集合: {COLLECTION_NAME}")
        
        # 获取所有.c文件
        all_files = get_c_files_recursive(RIOT_ROOT, TARGET_DIRS)
        logging.info(f"总共找到 {len(all_files)} 个.c文件")
        
        if not all_files:
            logging.warning("未找到任何.c文件")
            return
        
        # 批量处理文件
        processed_count, failed_count = process_files_in_batches(
            collection, all_files, BATCH_SIZE
        )
        
        # 输出统计信息
        logging.info(f"处理完成! 成功: {processed_count}, 失败: {failed_count}")
        logging.info(f"集合中现有文档数量: {collection.count()}")
        
    except Exception as e:
        logging.error(f"主程序执行失败: {str(e)}")
        raise

# 执行主函数
if __name__ == "__main__":
    main()
