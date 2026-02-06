from pymongo import MongoClient
from datetime import datetime

# Connect to local MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["med_gemma_app"]
chat_collection = db["conversations"]

def save_chat_turn(user_query, ai_response, metadata=None):
    """
    Saves a full exchange as a single document. 
    In NoSQL, we can nest the AI and User messages together.
    """
    document = {
        "timestamp": datetime.now(),
        "user_input": user_query,
        "ai_output": ai_response,
        "metadata": metadata or {
            "model": "med-gemma-2b",
            "quantization": "4-bit"
        }
    }
    return chat_collection.insert_one(document)

def get_all_history():
    """Returns all conversations sorted by time."""
    return list(chat_collection.find().sort("timestamp", 1))