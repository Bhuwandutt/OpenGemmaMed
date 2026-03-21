import os
from pymongo import MongoClient
from datetime import datetime

# In Docker the service is named "mongodb"; locally it stays localhost.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

client = MongoClient(MONGO_URI)
db = client["med_gemma_app"]
chat_collection = db["conversations"]


async def save_chat_turn(query: str, response: str, metadata: dict, user_id: str):
    """Saves a full exchange as a single document."""
    document = {
        "patient_id": user_id,
        "timestamp": datetime.now(),
        "user_input": query,      # ← consistent field name
        "output": response,        # ← consistent field name
        "metadata": metadata or {
            "model": "med-gemma-1.5-4b-it",
            "quantization": "4-bit",
        },
    }
    return chat_collection.insert_one(document)


async def get_patient_history(patient_id: str):
    """Retrieves all conversation documents for a specific patient."""
    cursor = chat_collection.find({"patient_id": patient_id}).sort("timestamp", 1)

    history = []
    for doc in cursor:
        history.append({
            "timestamp": doc["timestamp"],
            "query": doc["user_input"],    # ← was doc["query"] — field didn't exist
            "response": doc["output"],      # ← was doc["response"] — field didn't exist
            "metadata": doc.get("metadata", {}),
        })
    return history


def get_all_history():
    """Returns all conversations sorted by time."""
    return list(chat_collection.find().sort("timestamp", 1))