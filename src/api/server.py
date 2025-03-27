"""
FastAPI server implementation for BinaryLM.
"""
import os
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our tokenizer implementation
from src.tokenizer.bpe import BPETokenizer

# Create FastAPI app
app = FastAPI(title="BinaryLM API", description="API for BinaryLM transformer model")

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tokenizer with a small vocabulary for demonstration
# In a real application, we would load a pre-trained tokenizer
tokenizer = BPETokenizer(vocab_size=1000)

# Sample texts for demo purposes
sample_texts = [
    "Hello world!",
    "This is a simple tokenizer demonstration.",
    "BPE (Byte-Pair Encoding) creates subword tokens.",
    "Machine learning models use tokenizers to process text.",
    "Transformers are powerful neural network architectures."
]

# Train the tokenizer on our sample texts
tokenizer.train(sample_texts)

# Define API request/response models
class TokenizeRequest(BaseModel):
    """Request model for tokenization endpoint."""
    text: str
    show_vectors: Optional[bool] = False


class Token(BaseModel):
    """Model for a single token."""
    text: str
    id: int
    type: str
    vector: Optional[List[float]] = None


class TokenStep(BaseModel):
    """Model for a tokenization step."""
    stage: str
    tokens: List[Token]


class TokenizeResponse(BaseModel):
    """Response model for tokenization endpoint."""
    steps: List[TokenStep]


@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "BinaryLM API is running", "version": "0.1.0"}


@app.post("/api/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """
    Tokenize the input text and return the tokenization steps.
    
    This endpoint demonstrates the different stages of tokenization:
    1. Raw text
    2. Pre-tokenization (splitting by whitespace)
    3. Subword tokenization (BPE)
    4. Token IDs and embeddings
    """
    try:
        text = request.text.strip()
        show_vectors = request.show_vectors
        
        # Step 1: Raw text
        raw_step = TokenStep(
            stage="Raw Text",
            tokens=[Token(text=text, id=0, type="raw")]
        )
        
        # Step 2: Pre-tokenization (split by whitespace)
        pre_tokens = []
        for i, word in enumerate(text.split()):
            pre_tokens.append(Token(
                text=word,
                id=i,
                type="word"
            ))
        
        pre_tokenization_step = TokenStep(
            stage="Pre-tokenization",
            tokens=pre_tokens
        )
        
        # Step 3: Subword tokenization (simplified for demo)
        # In a real implementation, this would use the actual BPE algorithm
        subword_tokens = []
        token_idx = 0
        
        for i, word in enumerate(text.split()):
            # Simple heuristic: split words longer than 5 characters
            if len(word) > 5:
                prefix = word[:3]
                suffix = word[3:]
                subword_tokens.append(Token(
                    text=prefix,
                    id=token_idx,
                    type="subword-start"
                ))
                token_idx += 1
                subword_tokens.append(Token(
                    text=suffix,
                    id=token_idx,
                    type="subword-continuation"
                ))
                token_idx += 1
            else:
                subword_tokens.append(Token(
                    text=word,
                    id=token_idx,
                    type="word"
                ))
                token_idx += 1
        
        subword_step = TokenStep(
            stage="Subword Tokenization",
            tokens=subword_tokens
        )
        
        # Step 4: Convert to token IDs and generate vectors
        # Use our actual tokenizer to encode the text
        token_ids = tokenizer.encode(text)
        
        # Map these back to our tokens and generate random vectors if requested
        final_tokens = []
        for i, token in enumerate(subword_tokens):
            # Mock embedding vector for demonstration
            vector = None
            if show_vectors:
                import random
                # Generate 8-dimensional random vector with values between -1 and 1
                vector = [round(random.uniform(-1, 1), 2) for _ in range(8)]
            
            final_tokens.append(Token(
                text=token.text,
                id=1000 + i,  # Use a different ID range to show these are from the tokenizer
                type=token.type,
                vector=vector
            ))
        
        token_step = TokenStep(
            stage="Token IDs & Embeddings",
            tokens=final_tokens
        )
        
        # Return all steps in the tokenization process
        return TokenizeResponse(
            steps=[raw_step, pre_tokenization_step, subword_step, token_step]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tokenizer/info")
async def tokenizer_info():
    """Get information about the tokenizer."""
    return {
        "vocabulary_size": len(tokenizer.token_to_id),
        "special_tokens": list(tokenizer.special_tokens.values()),
        "algorithm": "BPE (Byte-Pair Encoding)",
        "sample_tokens": list(tokenizer.token_to_id.keys())[:20]  # Return first 20 tokens
    }


# Run the server with: uvicorn src.api.server:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 