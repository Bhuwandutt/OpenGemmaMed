import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class MedGemmaEngine:
    def __init__(self, model_id="google/medgemma-1.5-4b-it"):
        load_dotenv()
        self.token = os.getenv("HF_TOKEN")
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Current Device: {torch.cuda.get_device_name(0)}")
        else:
            # Check if drivers are even visible to the OS
            import subprocess
            try:
                subprocess.check_output('nvidia-smi')
                print("NVIDIA Drivers are installed but Torch cannot see them.")
            except:
                print("NVIDIA Drivers are NOT found by the system.")
    
        # Verify Hardware
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("WARNING: Running on CPU")

    def initialize(self):
        """Loads the tokenizer and model into VRAM."""
        print(f"Initializing {self.model_id}...")

        # Optimized 4-bit qunatization CachyOS + NVIDIA
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