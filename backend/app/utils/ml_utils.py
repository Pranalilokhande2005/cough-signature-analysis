import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class MLUtils:
    """Utility functions for machine learning operations"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray = None, 
                         class_names: List[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = np.mean(y_true == y_pred)
        
        # Classification report
        if class_names:
            metrics['classification_report'] = classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True
            )
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # AUC score (for binary classification)
        if y_proba is not None and len(np.unique(y_true)) == 2:
            metrics['auc_score'] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                            title: str = "Confusion Matrix") -> plt.Figure:
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], 
                            metrics: List[str] = ['accuracy', 'loss']) -> plt.Figure:
        """Plot training history"""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            axes[i].plot(history[metric], label=f'Training {metric}')
            axes[i].plot(history[f'val_{metric}'], label=f'Validation {metric}')
            axes[i].set_title(f'Model {metric.capitalize()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def set_random_seeds(seed: int = 42):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    @staticmethod
    def get_model_summary(model: tf.keras.Model) -> str:
        """Get model summary as string"""
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            model.summary()
        return f.getvalue()
    
    @staticmethod
    def calculate_feature_importance(model: tf.keras.Model, 
                                   X: np.ndarray, 
                                   y: np.ndarray, 
                                   feature_names: List[str] = None) -> Dict[str, float]:
        """Calculate feature importance using permutation importance"""
        # Get baseline score
        baseline_score = model.evaluate(X, y, verbose=0)[1]
        
        importances = {}
        
        for i in range(X.shape[1]):
            # Create a copy of X
            X_permuted = X.copy()
            
            # Permute the i-th feature
            np.random.shuffle(X_permuted[:, i])
            
            # Calculate score with permuted feature
            permuted_score = model.evaluate(X_permuted, y, verbose=0)[1]
            
            # Calculate importance
            importance = baseline_score - permuted_score
            feature_name = feature_names[i] if feature_names else f'feature_{i}'
            importances[feature_name] = importance
        
        return importances