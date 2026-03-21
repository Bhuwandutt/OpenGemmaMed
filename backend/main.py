from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from model import MedGemmaEngine
from database import save_chat_turn, get_patient_history
from threading import Thread

# --------------------------------------------------------------------------- #
# Lifespan: load model once at startup, clean up on shutdown
# --------------------------------------------------------------------------- #
engine = MedGemmaEngine() # __init__ is called to load model configuartion.


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.initialize()
    yield
    # teardown (if needed) goes here, runs after server shutdown



app = FastAPI(title="Med-Gemma API", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# CORS — allow the Vite dev server (port 5173) and any prod origin you add
# --------------------------------------------------------------------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:4173",   # Vite preview
        # Add your production domain here, e.g. "https://yourapp.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Query(BaseModel):
    prompt: str = Field(..., min_length=1, example="What are the symptoms of flu?")

# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
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
            user_id, 
            query.prompt, 
            full_response, 
            {"model": "medgemma-1.5-4b-it", "streamed": True}
        )
    return StreamingResponse(event_generator(), media_type="text/plain")

@app.get("/history/{patient_id}")
async def get_history(patient_id: str): 
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
