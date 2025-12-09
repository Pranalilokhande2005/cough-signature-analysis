import tensorflow as tf
import numpy as np
from pathlib import Path
import joblib
from typing import Optional, Dict, Any

class ModelManager:
    """Manages model loading, saving, and versioning"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.loaded_models = {}
        
    def load_model(self, model_name: str, model_path: Path) -> tf.keras.Model:
        """Load a model and cache it"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
            
        try:
            model = tf.keras.models.load_model(model_path)
            self.loaded_models[model_name] = model
            return model
        except Exception as e:
            raise Exception(f"Failed to load model {model_name}: {str(e)}")
    
    def save_model(self, model: tf.keras.Model, model_name: str, version: str = "latest"):
        """Save model with versioning"""
        model_path = self.models_dir / f"{model_name}_{version}.h5"
        model.save(model_path)
        return model_path
    
    def get_model_info(self, model: tf.keras.Model) -> Dict[str, Any]:
        """Get model information"""
        info = {
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "num_parameters": model.count_params(),
            "layers": len(model.layers)
        }
        return info
    
    def compare_models(self, model1: tf.keras.Model, model2: tf.keras.Model) -> Dict[str, Any]:
        """Compare two models"""
        comparison = {
            "architecture_match": model1.to_json() == model2.to_json(),
            "input_shape_match": model1.input_shape == model2.input_shape,
            "output_shape_match": model1.output_shape == model2.output_shape,
            "parameter_count_diff": abs(model1.count_params() - model2.count_params())
        }
        return comparison