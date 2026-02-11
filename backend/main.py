from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from model import MedGemmaEngine
from database import save_chat_turn, get_patient_history
import time

app = FastAPI(title="Med-Gemma API") # Create FastAPI instance called app

# Initialize Engine globally
engine = MedGemmaEngine() # __init__ is called to load model configuartion. 
engine.initialize() #Initilize function to load the model weights into memory


class Query(BaseModel):
    prompt: str = Field(..., min_length=1, example="What are the symptoms of flu?")

@app.post("/chat/{user_id}")
def ask_medgemma(user_id: str, query: Query):
    start_time = time.time() 
    
    # Generate response
    response = engine.generate_response(query.prompt)
    
    end_time = time.time()
    latency = round(end_time - start_time, 2)
    # Save to MongoDB
    metadata = {"latency": latency, "model": "med-gemma-1.5"}
    database.save_chat_turn(query.prompt, response, metadata,user_id)
    
    return {
        "user": user_id,
        "answer": response,
        "metadata": metadata
    }

@app.get("/history/{patient_id}")
async def get_history(patient_id: str): # Async helps run concurennt processes 
    history = await get_patient_history(patient_id) # await helps run the other function while we wait for response from Database. 

    if not history:
        return {"message": "No history found for this patient", "data": []}
    
    return {
        "patient_id": patient_id,
        "count": len(history),
        "data": history
    }

@app.get("/health")
def health_check():
    return {"status": "GPU Engine Active", "vram_usage": "check nvidia-smi"}