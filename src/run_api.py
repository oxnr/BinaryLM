#!/usr/bin/env python3
"""
Script to run the BinaryLM API server.
"""
import uvicorn

if __name__ == "__main__":
    # Run the FastAPI server with auto-reload for development
    uvicorn.run("api.server:app", host="0.0.0.0", port=5000, reload=True) 