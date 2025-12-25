from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

try:
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-02-15-preview"
    )
    
    completion = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
        ],
    )
    
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")