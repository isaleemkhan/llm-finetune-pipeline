"""
FastAPI inference server for the fine-tuned model.

Tries to use a vLLM backend for high throughput; if vLLM is unavailable it
falls back to a plain HuggingFace Transformers backend so the server still runs
anywhere. Exposes /generate, /chat, /model/info and /health.

Run:
    export MODEL_PATH=./merged
    uvicorn serving.server:app --host 0.0.0.0 --port 8000
"""

import os
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "./merged")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))

app = FastAPI(title="LLM Fine-tune Pipeline Server", version="1.0.0")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


class Backend:
    """Abstracts over vLLM and Transformers so the routes stay simple."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.kind = None
        self._load()

    def _load(self):
        try:
            from vllm import LLM, SamplingParams  # noqa: F401

            self.LLM = LLM
            self.SamplingParams = SamplingParams
            self.engine = LLM(model=self.model_path)
            self.kind = "vllm"
            print(f"[backend] vLLM loaded from {self.model_path}")
            return
        except Exception as exc:  # vLLM not installed or failed to load
            print(f"[backend] vLLM unavailable ({exc}); falling back to Transformers.")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()
        self.kind = "transformers"
        print(f"[backend] Transformers loaded from {self.model_path}")

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        if self.kind == "vllm":
            params = self.SamplingParams(
                max_tokens=max_new_tokens, temperature=temperature, top_p=top_p
            )
            out = self.engine.generate([prompt], params)
            return out[0].outputs[0].text

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def stream(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float):
        """Yield text chunks. Uses TextIteratorStreamer for the Transformers backend."""
        if self.kind == "vllm":
            yield self.generate(prompt, max_new_tokens, temperature, top_p)
            return

        from threading import Thread
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        for token in streamer:
            yield token
        thread.join()


backend: Optional[Backend] = None


def format_chat(messages: List[ChatMessage]) -> str:
    """Render chat messages into a single LLaMA 3 style prompt."""
    parts = ["<|begin_of_text|>"]
    for m in messages:
        parts.append(
            f"<|start_header_id|>{m.role}<|end_header_id|>\n{m.content}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    return "".join(parts)


@app.on_event("startup")
def _startup():
    global backend
    backend = Backend(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "backend": backend.kind if backend else "loading"}


@app.get("/model/info")
def model_info():
    if backend is None:
        raise HTTPException(status_code=503, detail="Model still loading.")
    return {
        "model_path": MODEL_PATH,
        "backend": backend.kind,
        "max_new_tokens_limit": 4096,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    if backend is None:
        raise HTTPException(status_code=503, detail="Model still loading.")
    if req.stream:
        gen = backend.stream(req.prompt, req.max_new_tokens, req.temperature, req.top_p)
        return StreamingResponse(gen, media_type="text/plain")
    start = time.time()
    text = backend.generate(req.prompt, req.max_new_tokens, req.temperature, req.top_p)
    return {"text": text, "latency_s": round(time.time() - start, 3)}


@app.post("/chat")
def chat(req: ChatRequest):
    if backend is None:
        raise HTTPException(status_code=503, detail="Model still loading.")
    prompt = format_chat(req.messages)
    if req.stream:
        gen = backend.stream(prompt, req.max_new_tokens, req.temperature, req.top_p)
        return StreamingResponse(gen, media_type="text/plain")
    start = time.time()
    text = backend.generate(prompt, req.max_new_tokens, req.temperature, req.top_p)
    return {
        "message": {"role": "assistant", "content": text},
        "latency_s": round(time.time() - start, 3),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
