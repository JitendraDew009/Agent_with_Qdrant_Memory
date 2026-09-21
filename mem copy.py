# pip install mem0ai openai python-dotenv qdrant-client
from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI
import os
import json

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# OpenAI wrapper initialization targeting Google's structural path
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_API_KEY
)

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "gemini",
        "config": {
            "api_key": GEMINI_API_KEY, 
            "model": "models/gemini-embedding-001",
            "embedding_dims": 768  # Enforces 768 native array dimension sizes
        }
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "api_key": GEMINI_API_KEY, 
            "model": "gemini-3.7-flash"  # Smooth and fast model layer
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost", 
            "port": 6333,
            "collection_name": "gemini_memories",
            "embedding_model_dims": 768  # Enforces matching database topology shape
        }
    },
    "graph_store":{
        "provider": "neo4j",
        "config" :{
            "url": NEO_CONNECTION_URI,
            "username": NEO_USERNAME,
            "password": NEO_PASSWORD

        }
    }
}

# The client will now build a brand new isolated index block inside Qdrant
mem_client = Memory.from_config(config)
while True:
    print("Ask the AI something: ")
    user_query = input()
    # retreive the memory
    search_memory = mem_client.search(query=user_query, filters={"user_id":"Jitendra"} )

    memories = [
        f"ID: {mem.get("id")}\nMemory : {mem.get("memory")}" 
        for mem in search_memory.get("results")
    ]

    print("Found Memories", memories)

    SYSTEM_PROMPT = f"""
        Here is the context about user:
        {json.dumps(memories)}
    """

    response = client.chat.completions.create(
        model='gemini-3.7-flash',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_query}
        ]
    )

    ai_response = response.choices[0].message.content
    print("AI: ", ai_response)

    # Mem0 processes the structural dialog and stores the cleanly converted vector
    mem_client.add(
        user_id='Jitendra',
        messages=[
            {'role': 'user', 'content': user_query},
            {'role': 'assistant', 'content': ai_response}
        ]
    )
    print("Memory has been saved successfully...")
