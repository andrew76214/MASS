import os
import getpass
from dotenv import load_dotenv

def load_and_prompt_env(var_names):
    # 載入 .env 檔案
    load_dotenv()

    for var in var_names:
        # 嘗試從環境取得變數
        value = os.getenv(var)
        if not value:
            # 若變數未定義，則提示使用者輸入
            value = getpass.getpass(f"Please provide your {var}: ")
            os.environ[var] = value
        
        # 確保 OPENAI_API_KEY 存在
        if var == "OPENAI_API_KEY" and not value:
            raise ValueError("OPENAI_API_KEY is required but not set.")
        
        # 確保 OPENAI_API_KEY 存在
        if var == "TAVILY_API_KEY" and not value:
            raise ValueError("TAVILY_API_KEY is required but not set.")

    print("Environment variables loaded successfully.")