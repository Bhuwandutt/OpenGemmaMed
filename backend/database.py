from pymongo import MongoClient
from datetime import datetime

# Connect to local MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["med_gemma_app"]
chat_collection = db["conversations"]

def save_chat_turn(query, response, metadata=None, patient_id):
    """
    Saves a full exchange as a single document.
    """
    document = {
        "patient_id": patient_id,
        "timestamp": datetime.now(),
        "user_input": query,
        "output": response,
        "metadata": metadata or {
            "model": "med-gemma-1.5-4b-it",
            "quantization": "4-bit"
        }
    }
    return chat_collection.insert_one(document)
    
def get_patient_history(patient_id: str):
    """Retrieves all conversation documents for a specific patient."""
    # Find all documents where patient_id matches, sorted by timestamp (oldest to newest)
    cursor = chat_collection.find({"patient_id": patient_id}).sort("timestamp", 1)
    
    # Convert MongoDB cursor to a list of dictionaries
    history = []
    for doc in cursor:
        history.append({
            "timestamp": doc["timestamp"],
            "query": doc["query"],
            "response": doc["response"],
            "metadata": doc.get("metadata", {})
        })
    return history

def get_all_history():
    """Returns all conversations sorted by time."""
    return list(chat_collection.find().sort("timestamp", 1))