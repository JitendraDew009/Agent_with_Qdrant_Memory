# pip install mem0ai
from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI
import os


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = client = OpenAI(
     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key= GEMINI_API_KEY
)

config = {
    "version": "v1.1",
    "embedder": {
        "provider" : "gemini",
        "config" : {"api_key": GEMINI_API_KEY, 'model': "models/gemini-embedding-001"},
        "embedding_dims": 768 
    },
    "llm" : {
        "provider" : "gemini",
        "config" : {"api_key": GEMINI_API_KEY, 'model': "gemini-3.6-flash"}
    },
    "vector_store": {
        "provider": "qdrant",
        "config" : {"host": "localhost", "port": 6333},
        "collection_name": "gemini_memories",
        "embedding_model_dims": 768
    }


}

mem_client = Memory.from_config(config)

print("Ask the AI something: ")

user_query = input()
response = client.chat.completions.create(
    model='gemini-3.6-flash',
    messages = [
        {'role': 'user', 'content' : user_query}
    ]
    
)

ai_response = response.choices[0].message.content
print("AI: ",ai_response)

mem_client.add(
    user_id= 'Jitendra',
    messages=[
        {'role': 'user', 'content' : user_query},
        {'role': 'assistant', 'content' : ai_response}

    ]
)
print("Memeory has been saved...")