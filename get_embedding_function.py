from langchain_ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
import os
from dotenv import load_dotenv


import requests

load_dotenv()
windows_ip = os.getenv("windows_ip")
url = f"http://{windows_ip}:11434/api/version"

try:
    response = requests.get(url)
    print(f"Connection successful: {response.text}")
except Exception as e:
    print(f"Connection failed: {e}")


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    windowsip=os.getenv("windows_ip")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=f"http://{windowsip}:11434")
    return embeddings
