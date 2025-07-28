"""
Model Server - FastAPI-based model serving application
Demonstrates Docker model runner capabilities
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Docker Model Runner",
    description="A containerized model serving API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    data: Union[List[List[float]], str]  # Support both numerical data and text
    model_name: Optional[str] = "default"

class TextGenerationRequest(BaseModel):
    prompt: str
    model_name: Optional[str] = "llama3.2"
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7

class PredictionResponse(BaseModel):
    predictions: Union[List[float], str]  # Support both numerical and text predictions
    model_name: str
    timestamp: str
    processing_time_ms: float

class TextGenerationResponse(BaseModel):
    generated_text: str
    model_name: str
    timestamp: str
    processing_time_ms: float
    prompt: str

class ModelInfo(BaseModel):
    name: str
    type: str
    loaded: bool
    version: str
    description: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int
    uptime_seconds: float

# Global variables for tracking
start_time = datetime.now()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting Docker Model Runner...")
    await model_manager.load_default_models()
    logger.info("Model Runner started successfully")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Docker Model Runner API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - start_time).total_seconds()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(model_manager.get_loaded_models()),
        uptime_seconds=uptime
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models"""
    return model_manager.get_model_info()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the specified model"""
    start_time_pred = datetime.now()
    
    try:
        # Validate input data
        if not request.data:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Handle different input types for different models
        if request.model_name == "llama3.2" or isinstance(request.data, str):
            # For Llama model, expect text input
            if isinstance(request.data, str):
                input_data = request.data
            else:
                raise HTTPException(status_code=400, detail="Llama model requires text input")
        else:
            # For traditional ML models, expect numerical data
            if isinstance(request.data, str):
                raise HTTPException(status_code=400, detail="Traditional ML models require numerical data")
            input_data = np.array(request.data)
        
        # Make prediction
        predictions = await model_manager.predict(
            data=input_data,
            model_name=request.model_name
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time_pred).total_seconds() * 1000
        
        return PredictionResponse(
            predictions=predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            model_name=request.model_name,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generate text using Llama model"""
    start_time_pred = datetime.now()
    
    try:
        # Validate model
        if request.model_name not in model_manager.get_loaded_models():
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not loaded")
        
        # Generate text
        generated_text = await model_manager.predict(
            data=request.prompt,
            model_name=request.model_name
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time_pred).total_seconds() * 1000
        
        return TextGenerationResponse(
            generated_text=generated_text,
            model_name=request.model_name,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
            prompt=request.prompt
        )
        
    except Exception as e:
        logger.error(f"Text generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a specific model"""
    try:
        success = await model_manager.load_model(model_name)
        if success:
            return {"message": f"Model {model_name} loaded successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a specific model"""
    try:
        success = await model_manager.unload_model(model_name)
        if success:
            return {"message": f"Model {model_name} unloaded successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")
    except Exception as e:
        logger.error(f"Model unloading error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

@app.post("/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a new model file"""
    try:
        # Save uploaded file
        model_path = f"/app/models/{file.filename}"
        with open(model_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {"message": f"Model {file.filename} uploaded successfully"}
        
    except Exception as e:
        logger.error(f"Model upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
