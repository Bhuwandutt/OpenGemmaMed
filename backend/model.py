import os
import re
import torch
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread
from typing import Generator, Tuple


# --------------------------------------------------------------------------- #
#  Why the previous token-by-token filter failed
#  -----------------------------------------------
#  TextIteratorStreamer yields *decoded text* for each new token id, not raw
#  token ids.  A single tag like <unused94> can decode as:
#    chunk 1 → "<"
#    chunk 2 → "un"
#    chunk 3 → "used"
#    chunk 4 → "94>"
#  So buffer.find("<unused94>") never returns a match mid-stream.
#
#  Fix: accumulate the full raw text in one string, apply a regex that strips
#  ALL thought blocks in one pass, then diff against how much clean text we
#  have already yielded.  This is O(n) per token but n is small, and it is
#  100 % correct regardless of how the tokenizer splits tags.
# --------------------------------------------------------------------------- #

# Matches <unusedN>, <thought>, <thinking> blocks and everything inside them.
# The alternation covers all known Gemma reasoning-token wrappers.
_THOUGHT_RE = re.compile(
    r"<(?:unused\d+|thought|thinking)>.*?</(?:unused\d+|thought|thinking)>",
    re.DOTALL | re.IGNORECASE,
)

# Also strip bare opening tags that were never closed (truncated thought block)
_UNCLOSED_RE = re.compile(
    r"<(?:unused\d+|thought|thinking)>.*$",
    re.DOTALL | re.IGNORECASE,
)


def _clean(text: str) -> str:
    """Remove all thought blocks from a text string."""
    text = _THOUGHT_RE.sub("", text)
    text = _UNCLOSED_RE.sub("", text)   # remove any unclosed block at the end
    return text


# --------------------------------------------------------------------------- #
#  MedGemmaEngine
# --------------------------------------------------------------------------- #

class MedGemmaEngine:
    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it"):
        load_dotenv()
        self.token = os.getenv("HF_TOKEN")
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("RUNNING ON CUDA GPU" if self.device == "cuda" else "WARNING: Running on CPU")

    # ------------------------------------------------------------------ #
    def initialize(self):
        """Loads the tokenizer and model into VRAM."""
        print(f"Initializing {self.model_id}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=self.token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.token,
        )
        print("Model loaded successfully.")

    # ------------------------------------------------------------------ #
    def _run_inference(self, prompt: str, max_tokens: int) -> Tuple[str, bool]:
        """
        Runs non-streaming inference and returns (clean_text, was_truncated).
        Used both for direct responses and for the summarisation fallback.
        """
        formatted = (
            f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
            )

        new_tokens = output_ids.shape[1] - input_len
        was_truncated = (new_tokens >= max_tokens)

        raw = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # decode() includes the prompt text — strip it
        # (the prompt ends with "<start_of_turn>model\n")
        marker = "<start_of_turn>model\n"
        if marker in raw:
            raw = raw.split(marker)[-1]

        return _clean(raw).strip(), was_truncated

    # ------------------------------------------------------------------ #
    def stream_response(
        self,
        prompt: str,
        max_tokens: int = 1024,
    ) -> Generator[str, None, None]:
        """
        Yields clean text chunks to the caller.

        Strategy
        --------
        1. Run generation in a background thread via TextIteratorStreamer.
        2. Accumulate the *raw* decoded text as tokens arrive.
        3. After every token, apply _clean() to the full accumulated string
           and yield only the *new* portion (the diff).
        4. If truncation is detected, run a short summarisation inference
           and append a clean one-paragraph summary + notice.
        """
        if not self.model:
            raise RuntimeError("Model not initialized.")

        formatted = (
            f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        raw_streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = dict(
            **inputs,
            streamer=raw_streamer,
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        raw_accumulator = ""   # every raw chunk appended here
        yielded_len = 0        # how many chars of *clean* text already sent
        token_count = 0

        for raw_chunk in raw_streamer:
            token_count += 1
            raw_accumulator += raw_chunk

            # Clean the FULL accumulated string — regex handles split tags
            clean_so_far = _clean(raw_accumulator)

            # Yield only the new clean portion
            new_text = clean_so_far[yielded_len:]
            if new_text:
                yield new_text
                yielded_len += len(new_text)

        thread.join()

        # ── Truncation recovery ──────────────────────────────────────────
        was_truncated = (token_count >= max_tokens)
        if was_truncated:
            # Build a context-aware summary prompt using the clean response
            clean_response = _clean(raw_accumulator).strip()
            context_tail = clean_response[-800:] if len(clean_response) > 800 else clean_response

            summary_prompt = (
                f"The following is a medical explanation that was cut off mid-way "
                f"due to length limits:\n\n\"{context_tail}\"\n\n"
                f"Write a single concise paragraph (3-5 sentences) that cleanly "
                f"summarises and concludes the key points from that explanation. "
                f"Start directly with the summary, no preamble."
            )

            yield "\n\n---\n**Summary:** "
            summary, _ = self._run_inference(summary_prompt, max_tokens=200)
            yield summary

    # ------------------------------------------------------------------ #
    def generate_response(self, prompt: str, max_tokens: int = 1024) -> str:
        """Non-streaming convenience wrapper."""
        text, _ = self._run_inference(prompt, max_tokens)
        return text
