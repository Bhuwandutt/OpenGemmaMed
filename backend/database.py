from pymongo import MongoClient
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv


load_dotenv()
# Set MONGO_URI in .env for auth
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")


# Connect to local MongoDB
client = AsyncIOMotorClient(MONGO_URI)
db = client["med_gemma_app"]
chat_collection = db["conversations"]

async def save_chat_turn(user_id: str, query: str, response: str, metadata: dict = None):
    """
    Saves a full exchange as a single document.
    Field names match what get_patient_history reads back.
    """
    document = {
        "patient_id": user_id,
        "timestamp": datetime.now(timezone.utc),
        "user_input": query,
        "output": response,
        "metadata": metadata or {
            "model": "med-gemma-1.5-4b-it",
            "quantization": "4-bit"
        }
    }
    result = await chat_collection.insert_one(document)
    return result

    
async def get_patient_history(patient_id: str):
    """Retrieves all conversation documents for a specific patient."""
    # Find all documents where patient_id matches, sorted by timestamp (oldest to newest)
    cursor = chat_collection.find({"patient_id": patient_id},{"_id":0}).sort("timestamp", 1) # exclude Mongo's internal _id from response
    
    # Convert MongoDB cursor to a list of dictionaries
    history = []
    async for doc in cursor:
        history.append({
            "timestamp": doc["timestamp"],
            "query": doc["user_input"],   # mactches save key
            "response": doc["output"],
            "metadata": doc.get("metadata", {})
        })
    return history

async def get_all_history():
    """Returns all conversations sorted by time."""

    cursor = chat_collection.find({}, {"_id": 0}).sort("timestamp", 1)
    return [doc async for doc in cursor]