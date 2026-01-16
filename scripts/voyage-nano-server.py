#!/usr/bin/env python3
"""
Voyage-4-Nano Local Embedding Server

Runs voyage-4-nano locally and exposes an OpenAI-compatible embedding API.
This allows qmd to use voyage-4-nano for local query embeddings while
keeping voyage-4-large embeddings for documents (asymmetric retrieval).

Usage:
    python scripts/voyage-nano-server.py
    
Then set:
    export QMD_PROVIDER=openai
    export OPENAI_API_BASE=http://localhost:8765/v1
    export OPENAI_EMBED_MODEL=voyage-4-nano
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoModel, AutoConfig, AutoTokenizer, PreTrainedModel
from typing import List
import argparse

# Model setup
MODEL_NAME = "voyageai/voyage-4-nano"
DEFAULT_PORT = 8765
DEFAULT_DIM = 1024

class Pooling(nn.Module):
    def forward(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

class Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)

class AutoModelForEmbedding(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.vocab_size = config.vocab_size
        self.linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.normalize = Normalize()
        self.pooling = Pooling()

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = self.linear(outputs.last_hidden_state)
        return self.normalize(self.pooling(hidden_states, attention_mask))


# Global model instance
model = None
tokenizer = None
device = None

def load_model():
    global model, tokenizer, device
    
    print(f"Loading {MODEL_NAME}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForEmbedding.from_pretrained(MODEL_NAME, config=config, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded! Ready to serve embeddings.")

def get_embeddings(texts: List[str], dimensions: int = DEFAULT_DIM) -> List[List[float]]:
    """Get embeddings for a list of texts."""
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32000,
            return_tensors="pt"
        ).to(device)
        
        embeddings = model(**inputs)
        
        # Truncate to requested dimensions (Matryoshka)
        if dimensions < embeddings.shape[1]:
            embeddings = embeddings[:, :dimensions]
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()


class EmbeddingHandler(BaseHTTPRequestHandler):
    def _send_response(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_POST(self):
        if self.path == "/v1/embeddings":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
            
            texts = body.get("input", [])
            if isinstance(texts, str):
                texts = [texts]
            
            dimensions = body.get("dimensions", DEFAULT_DIM)
            
            try:
                embeddings = get_embeddings(texts, dimensions)
                
                response = {
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "embedding": emb,
                            "index": i
                        }
                        for i, emb in enumerate(embeddings)
                    ],
                    "model": "voyage-4-nano",
                    "usage": {
                        "prompt_tokens": sum(len(t.split()) for t in texts),
                        "total_tokens": sum(len(t.split()) for t in texts)
                    }
                }
                self._send_response(response)
            except Exception as e:
                self._send_response({"error": str(e)}, 500)
        else:
            self._send_response({"error": "Not found"}, 404)
    
    def do_GET(self):
        if self.path == "/health":
            self._send_response({"status": "ok", "model": MODEL_NAME})
        else:
            self._send_response({"error": "Not found"}, 404)
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="Voyage-4-Nano Embedding Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()
    
    load_model()
    
    server = HTTPServer((args.host, args.port), EmbeddingHandler)
    print(f"\n🚀 Voyage-4-Nano server running at http://{args.host}:{args.port}")
    print(f"   Endpoint: POST /v1/embeddings")
    print(f"   Health:   GET /health")
    print(f"\nTo use with qmd:")
    print(f"   export QMD_PROVIDER=openai")
    print(f"   export OPENAI_API_BASE=http://{args.host}:{args.port}/v1")
    print(f"   export OPENAI_EMBED_MODEL=voyage-4-nano")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
