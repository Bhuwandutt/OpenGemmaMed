import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class MedGemmaEngine: # currenty using MedGemma but this class can be scaled to any generic AI model. We can pass the model ID as argument. (To implement lpug and play AI model)
    
    def __init__(self, model_id="google/medgemma-1.5-4b-it"): #Fuction call after every instance creation

        load_dotenv()
        self.token = os.getenv("HF_TOKEN")
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        
        # Verify Hardware
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("WARNING: Running on CPU")
        else:
            print("RUNNING ON CUDA GPU")

    def initialize(self):
        """Loads the tokenizer and model into VRAM."""
        print(f"Initializing {self.model_id}...")

        # Optimized 4-bit quantization 
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.token)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.token
        )
        print("Model loaded successfully.")

    # def stream_response(self, prompt):
    #     """Generates tokens one by one for the UI."""
    #     inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
    #     streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
    #     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512, temperature=0.2)
        
    #     # We run generation in a background thread so the UI doesn't lock up
    #     thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
    #     thread.start()
        
    #     return streamer
    
    def generate_response(self, prompt, max_tokens=300):
        """Generates medical insights based on user input."""
        if not self.model:
            raise RuntimeError("Model not initialized. Call .initialize() first.")

        # Structured prompt for better clinical reasoning
        formatted_prompt = (
            f"<start_of_turn>user\n"
            f"You are a medical assistant. Use professional tone.\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.2, # Keeps it factual
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)