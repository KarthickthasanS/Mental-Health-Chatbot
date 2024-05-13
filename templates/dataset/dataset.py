
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import tensorflow as tf
# pip install transformers accelerate

from transformers import AutoTokenizer
from transformers import pipeline, AutoTokenizer
import transformers
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

# Load pipeline and tokenizer once (optimization)
model_name = "KarthickthasanS/MindMate"  # Update with your desired model name
pipeline = pipeline("text-generation", model=model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def calling(prompt: str):
  sequences = pipeline(
      f'[INST] {prompt} [/INST]',
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      max_length=200,
  )
  return sequences

app = FastAPI()
class TextInput(BaseModel):
    inputs: str
    

@app.get("/")
def status_gpu_check() -> dict[str, str]:
    gpu_msg = "Available" if tf.test.is_gpu_available() else "Unavailable"
    return {"status": "I am ALIVE!", "gpu": gpu_msg}

@app.post("/generate/")
async def generate_text(data: TextInput) -> dict[str, str]:
    try:
        params = data.parameters or {}
        response = calling(prompt=data.inputs)
        generated_texts = []
    
        for seq in response:
            generated_text = seq['generated_text']
            generated_texts.append(generated_text)
            print(f"Result: {generated_text}")
    return {"generated_texts": generated_texts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))