import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import OPENAI_BASE_URL, OPENAI_KEY

print(OPENAI_BASE_URL)