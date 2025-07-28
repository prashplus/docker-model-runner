"""
Model Manager - Handles loading, unloading, and managing ML models including Llama 3.2
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import asyncio

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages multiple ML models including Llama 3.2"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict] = {}
        self.models_dir = "/app/models"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info(f"Model Manager initialized with device: {self.device}")
    
    async def load_default_models(self):
        """Load default demo models including Llama 3.2"""
        logger.info("Loading default models...")
        
        # Create a simple demo model
        await self.create_demo_model()
        
        # Load Llama 3.2 model
        await self.load_llama_model()
        
        # Load any existing models from disk
        await self.load_models_from_disk()
    
    async def load_llama_model(self):
        """Load Llama 3.2 model"""
        try:
            logger.info("Loading Llama 3.2 model...")
            
            model_name = "meta-llama/Llama-3.2-1B-Instruct"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                pad_token="<|pad|>"
            )
            
            # Load model with appropriate settings for resource efficiency
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            # Create text generation pipeline
            text_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Store model and tokenizer
            self.models["llama3.2"] = text_pipeline
            self.tokenizers["llama3.2"] = tokenizer
            self.model_info["llama3.2"] = {
                "name": "llama3.2",
                "type": "LlamaForCausalLM",
                "loaded": True,
                "version": "3.2-1B-Instruct",
                "description": "Meta Llama 3.2 1B Instruct model for text generation",
                "framework": "transformers",
                "model_size": "1B parameters"
            }
            
            logger.info("Llama 3.2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Llama 3.2 model: {str(e)}")
            # Don't fail completely, just log the error
            pass
    
    async def create_demo_model(self):
        """Create a demo RandomForest model"""
        try:
            # Generate sample data
            X, y = make_classification(
                n_samples=1000,
                n_features=4,
                n_informative=3,
                n_redundant=1,
                n_classes=2,
                random_state=42
            )
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Store model
            self.models["default"] = model
            self.model_info["default"] = {
                "name": "default",
                "type": "RandomForestClassifier",
                "loaded": True,
                "version": "1.0.0",
                "description": "Demo binary classification model",
                "framework": "scikit-learn"
            }
            
            # Save to disk
            model_path = os.path.join(self.models_dir, "default_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            logger.info("Demo model created and loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to create demo model: {str(e)}")
    
    async def load_models_from_disk(self):
        """Load models from the models directory"""
        try:
            if not os.path.exists(self.models_dir):
                return
            
            for filename in os.listdir(self.models_dir):
                if filename.endswith('.pkl'):
                    model_name = filename.replace('.pkl', '').replace('_model', '')
                    if model_name not in self.models:
                        await self.load_model_from_file(filename, model_name)
                        
        except Exception as e:
            logger.error(f"Failed to load models from disk: {str(e)}")
    
    async def load_model_from_file(self, filename: str, model_name: str):
        """Load a model from a pickle file"""
        try:
            file_path = os.path.join(self.models_dir, filename)
            with open(file_path, "rb") as f:
                model = pickle.load(f)
            
            self.models[model_name] = model
            self.model_info[model_name] = {
                "name": model_name,
                "type": type(model).__name__,
                "loaded": True,
                "version": "1.0.0",
                "description": f"Model loaded from {filename}",
                "framework": "scikit-learn"
            }
            
            logger.info(f"Model {model_name} loaded from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {filename}: {str(e)}")
            return False
    
    async def load_model(self, model_name: str) -> bool:
        """Load a specific model"""
        try:
            if model_name in self.models:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            # Special handling for Llama model
            if model_name.lower() in ["llama", "llama3.2", "llama-3.2"]:
                await self.load_llama_model()
                return "llama3.2" in self.models
            
            # Try to load from file
            filename = f"{model_name}_model.pkl"
            return await self.load_model_from_file(filename, model_name)
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a specific model"""
        try:
            if model_name in self.models:
                # Special cleanup for transformer models
                if model_name == "llama3.2":
                    # Clear CUDA cache if using GPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                del self.models[model_name]
                if model_name in self.tokenizers:
                    del self.tokenizers[model_name]
                if model_name in self.model_info:
                    self.model_info[model_name]["loaded"] = False
                    
                logger.info(f"Model {model_name} unloaded")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {str(e)}")
            return False
    
    async def predict(self, data: Union[np.ndarray, List, str], model_name: str = "default") -> Union[np.ndarray, str, List]:
        """Make predictions using the specified model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not loaded")
            
            model = self.models[model_name]
            
            # Handle Llama model predictions (text generation)
            if model_name == "llama3.2":
                if isinstance(data, str):
                    prompt = data
                elif isinstance(data, list) and len(data) > 0:
                    prompt = str(data[0]) if not isinstance(data[0], str) else data[0]
                else:
                    raise ValueError("For Llama model, data should be a string prompt")
                
                # Format prompt for instruction model
                formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                
                # Generate response
                result = model(
                    formatted_prompt,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=model.tokenizer.eos_token_id
                )
                
                # Extract generated text
                generated_text = result[0]['generated_text']
                # Remove the prompt from the response
                response = generated_text.replace(formatted_prompt, "").strip()
                
                return response
            
            # Handle traditional ML models
            else:
                if isinstance(data, str):
                    raise ValueError("Traditional ML models require numerical data")
                
                # Convert to numpy array for traditional models
                input_data = np.array(data)
                
                # Make prediction
                if hasattr(model, 'predict_proba'):
                    # For classification models, return probabilities
                    predictions = model.predict_proba(input_data)[:, 1]  # Probability of positive class
                else:
                    predictions = model.predict(input_data)
                
                return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names"""
        return list(self.models.keys())
    
    def get_model_info(self) -> List[Dict]:
        """Get information about all models"""
        return list(self.model_info.values())
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded"""
        return model_name in self.models
