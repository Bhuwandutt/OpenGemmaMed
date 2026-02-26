from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from model import MedGemmaEngine
from database import save_chat_turn, get_patient_history
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer
from threading import Thread

import json 
import time

from fastapi.responses import StreamingResponse


app = FastAPI(title="Med-Gemma API") # Create FastAPI instance called app

# Initialize Engine globally
engine = MedGemmaEngine() # __init__ is called to load model configuartion. 
engine.initialize() #Initilize function to load the model weights into memory


class Query(BaseModel):
    prompt: str = Field(..., min_length=1, example="What are the symptoms of flu?")

@app.post("/ask/{user_id}")
async def ask_medgemma(user_id: str, query: Query, background_tasks: BackgroundTasks):
    streamer = engine.stream_response(query.prompt)

    def event_generator():
        full_response = ""
        for token in streamer:
            full_response += token
            yield token
        
        # Once the loop ends, the stream is finished!
        # Use a BackgroundTask so the user doesn't wait for the DB write
        background_tasks.add_task(
            save_chat_turn, 
            query.prompt, 
            full_response, 
            {"model": "med-gemma-1.5", "streamed": True}, 
            user_id
        )
    return StreamingResponse(event_generator(), media_type="text/plain")

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