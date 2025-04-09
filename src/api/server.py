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


class DocResponse(BaseModel):
    """Response model for documentation endpoint."""
    content: str
    filename: str


@app.get("/")
async def root():
    """Root endpoint, returns basic server info."""
    return {
        "name": "BinaryLM API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": [
            "/api/tokenize",
            "/api/tokenizer/info",
            "/api/docs/{filename}"
        ]
    }


@app.get("/api/docs/{filename}", response_model=DocResponse)
async def get_documentation(filename: str):
    """
    Serves documentation files like README.md, Glossary.md, etc.
    
    Args:
        filename: Name of the documentation file to fetch
        
    Returns:
        DocResponse: The content of the documentation file
    """
    # Get the project root directory (two levels up from this file)
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Try to find the file in the project root first
    file_path = os.path.join(current_dir, filename)
    
    # If not in root, check web/public directory
    if not os.path.exists(file_path):
        file_path = os.path.join(current_dir, "web", "public", filename)
    
    # If still not found, raise an error
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Documentation file '{filename}' not found")
    
    # Read the file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        return DocResponse(content=content, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@app.post("/api/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """
    Tokenize input text and return the step-by-step process.
    
    This endpoint demonstrates how the tokenization process works by showing
    each step from raw text to final tokens.
    
    Args:
        request: TokenizeRequest object containing text to tokenize
        
    Returns:
        TokenizeResponse: Step-by-step tokenization process
    """
    text = request.text
    show_vectors = request.show_vectors
    
    # In a real implementation, we would use our actual tokenizer
    # For now, we'll create a simplified demonstration
    
    steps = []
    
    # Step 1: Normalization (lowercase, remove extra spaces)
    normalized_text = text.lower().strip()
    while "  " in normalized_text:
        normalized_text = normalized_text.replace("  ", " ")
    
    normalization_tokens = [
        Token(
            text=text,
            id=-1,  # No ID for raw text
            type="raw"
        ),
        Token(
            text=normalized_text,
            id=-1,  # No ID for normalized text
            type="normalized"
        )
    ]
    
    steps.append(TokenStep(
        stage="normalization",
        tokens=normalization_tokens
    ))
    
    # Step 2: Pre-tokenization (split by whitespace)
    pre_tokens = normalized_text.split()
    pre_tokenization_tokens = [
        Token(
            text=token,
            id=-1,  # No ID yet
            type="pre_token"
        ) for token in pre_tokens
    ]
    
    steps.append(TokenStep(
        stage="pre_tokenization",
        tokens=pre_tokenization_tokens
    ))
    
    # Step 3: BPE tokenization 
    # For demo purposes, we'll just do some basic subword splitting
    bpe_tokens = tokenizer.tokenize(normalized_text)
    
    # Convert to Token objects
    bpe_tokenization_tokens = []
    for i, token in enumerate(bpe_tokens):
        token_obj = Token(
            text=token,
            id=i,  # Simple sequential ID for demo
            type="token"
        )
        
        # Add vector representation if requested
        if show_vectors:
            # In a real implementation, we would get actual embeddings
            # For the demo, we'll just use random values
            import random
            token_obj.vector = [random.uniform(-1, 1) for _ in range(4)]  # 4D vectors for demo
            
        bpe_tokenization_tokens.append(token_obj)
    
    steps.append(TokenStep(
        stage="bpe_tokenization",
        tokens=bpe_tokenization_tokens
    ))
    
    return TokenizeResponse(steps=steps)


@app.get("/api/tokenizer/info")
async def tokenizer_info():
    """
    Return information about the tokenizer.
    
    Returns:
        Dict: Information about the tokenizer configuration
    """
    return {
        "type": "BPE (Byte-Pair Encoding)",
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": {
            "PAD": "[PAD]",
            "UNK": "[UNK]",
            "BOS": "[BOS]",
            "EOS": "[EOS]"
        },
        "sample_vocabulary": list(tokenizer.vocab.keys())[:20] if hasattr(tokenizer, 'vocab') else [],
        "training_corpus_size": len(sample_texts),
        "max_token_length": tokenizer.max_token_length if hasattr(tokenizer, 'max_token_length') else None
    }


# Run the server with: uvicorn src.api.server:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 