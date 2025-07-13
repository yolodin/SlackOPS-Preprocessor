"""
ML-based classification module for detecting intent in Slack threads.
"""

import os
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, pipeline
)
from sentence_transformers import SentenceTransformer
import joblib

# Fallback to rule-based classification if ML fails
import classify


class MLIntentClassifier:
    """
    Machine Learning-based intent classifier using transformer models.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the ML classifier.
        
        Args:
            model_name: Hugging Face model name to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.label_mapping = {
            0: "bug_report",
            1: "feature_request", 
            2: "how_to_question",
            3: "troubleshooting",
            4: "configuration",
            5: "discussion",
            6: "announcement",
            7: "general"
        }
        self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
    def prepare_training_data(self, slack_threads: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        Prepare training data from Slack threads.
        
        Args:
            slack_threads: List of thread data dictionaries
            
        Returns:
            Tuple of (texts, labels)
        """
        texts = []
        labels = []
        
        for thread in slack_threads:
            # Format thread text
            formatted_text = self._format_thread_for_ml(thread)
            
            # Get rule-based classification as initial label
            # In practice, you'd want human-annotated labels
            rule_based_intent = classify.detect_intent(formatted_text)
            
            texts.append(formatted_text)
            labels.append(rule_based_intent)
            
        return texts, labels
    
    def _format_thread_for_ml(self, thread_data: Dict[str, Any]) -> str:
        """
        Format thread data for ML processing.
        
        Args:
            thread_data: Thread data dictionary
            
        Returns:
            Formatted text string
        """
        if not thread_data or 'messages' not in thread_data:
            return ""
            
        messages = thread_data['messages']
        text_parts = []
        
        for message in messages:
            user = message.get('user', 'user')
            text = message.get('text', '').strip()
            if text:
                text_parts.append(f"{user}: {text}")
        
        return " [SEP] ".join(text_parts)
    
    def train_model(self, texts: List[str], labels: List[str], 
                   test_size: float = 0.2, save_path: str = "models/intent_classifier"):
        """
        Train the ML model.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            test_size: Fraction of data to use for testing
            save_path: Path to save the trained model
        """
        # Create models directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert labels to numeric
        numeric_labels = [self.reverse_label_mapping.get(label, 7) for label in labels]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, numeric_labels, test_size=test_size, random_state=42, stratify=numeric_labels
        )
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.label_mapping)
        )
        
        # Tokenize data
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=512)
        test_encodings = self.tokenizer(X_test, truncation=True, padding=True, max_length=512)
        
        # Create dataset class
        class SlackDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
        
        train_dataset = SlackDataset(train_encodings, y_train)
        test_dataset = SlackDataset(test_encodings, y_test)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=save_path,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{save_path}/logs",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        
        # Train model
        print("Training ML intent classifier...")
        trainer.train()
        
        # Save model
        trainer.save_model(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Evaluate model
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Convert back to labels for reporting
        y_test_labels = [self.label_mapping[label] for label in y_test]
        y_pred_labels = [self.label_mapping[pred] for pred in y_pred]
        
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_labels))
        
        # Save label mappings
        with open(f"{save_path}/label_mapping.json", "w") as f:
            json.dump(self.label_mapping, f)
            
        print(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Load label mapping
            with open(f"{model_path}/label_mapping.json", "r") as f:
                self.label_mapping = {int(k): v for k, v in json.load(f).items()}
                self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
            
            # Create pipeline for inference
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=True
            )
            
            print(f"Model loaded from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to rule-based classification")
            self.pipeline = None
    
    def predict(self, formatted_text: str, thread_metadata: Dict[str, Any] = None) -> Tuple[str, float]:
        """
        Predict intent using the ML model.
        
        Args:
            formatted_text: Formatted thread text
            thread_metadata: Optional metadata about the thread
            
        Returns:
            Tuple of (predicted_intent, confidence_score)
        """
        if not self.pipeline:
            # Fallback to rule-based classification
            intent = classify.detect_intent(formatted_text, thread_metadata)
            confidence = classify.get_confidence_score(formatted_text, intent)
            return intent, confidence
        
        try:
            # Get predictions
            results = self.pipeline(formatted_text)
            
            # Find the highest scoring prediction
            best_prediction = max(results, key=lambda x: x['score'])
            
            # Map label back to intent name
            label_id = int(best_prediction['label'].split('_')[-1])
            intent = self.label_mapping.get(label_id, "general")
            confidence = best_prediction['score']
            
            return intent, confidence
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            # Fallback to rule-based classification
            intent = classify.detect_intent(formatted_text, thread_metadata)
            confidence = classify.get_confidence_score(formatted_text, intent)
            return intent, confidence


class SentenceEmbeddingClassifier:
    """
    Alternative lightweight classifier using sentence embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentence embedding classifier.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.classifier = None
        self.label_encoder = None
    
    def train_lightweight_model(self, texts: List[str], labels: List[str], 
                               save_path: str = "models/lightweight_classifier"):
        """
        Train a lightweight classifier using sentence embeddings.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            save_path: Path to save the trained model
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Create models directory
        os.makedirs(save_path, exist_ok=True)
        
        # Generate embeddings
        print("Generating sentence embeddings...")
        embeddings = self.embedding_model.encode(texts)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Train classifier
        print("Training lightweight classifier...")
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Lightweight model accuracy: {accuracy:.4f}")
        
        # Save model
        joblib.dump(self.classifier, f"{save_path}/classifier.pkl")
        joblib.dump(self.label_encoder, f"{save_path}/label_encoder.pkl")
        
        print(f"Lightweight model saved to {save_path}")
    
    def load_lightweight_model(self, model_path: str):
        """
        Load the lightweight model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            self.classifier = joblib.load(f"{model_path}/classifier.pkl")
            self.label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")
            print(f"Lightweight model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading lightweight model: {e}")
            self.classifier = None
            self.label_encoder = None
    
    def predict_lightweight(self, formatted_text: str) -> Tuple[str, float]:
        """
        Predict using the lightweight model.
        
        Args:
            formatted_text: Formatted thread text
            
        Returns:
            Tuple of (predicted_intent, confidence_score)
        """
        if not self.classifier or not self.label_encoder:
            # Fallback to rule-based classification
            intent = classify.detect_intent(formatted_text)
            confidence = classify.get_confidence_score(formatted_text, intent)
            return intent, confidence
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([formatted_text])
            
            # Predict
            prediction = self.classifier.predict(embedding)[0]
            probabilities = self.classifier.predict_proba(embedding)[0]
            
            # Get intent name and confidence
            intent = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            return intent, confidence
            
        except Exception as e:
            print(f"Error in lightweight prediction: {e}")
            # Fallback to rule-based classification
            intent = classify.detect_intent(formatted_text)
            confidence = classify.get_confidence_score(formatted_text, intent)
            return intent, confidence


# Global model instances
_ml_classifier = None
_lightweight_classifier = None


def initialize_ml_models(use_lightweight: bool = False):
    """
    Initialize ML models for classification.
    
    Args:
        use_lightweight: Whether to use lightweight embedding-based classifier
    """
    global _ml_classifier, _lightweight_classifier
    
    if use_lightweight:
        _lightweight_classifier = SentenceEmbeddingClassifier()
        _lightweight_classifier.load_lightweight_model("models/lightweight_classifier")
    else:
        _ml_classifier = MLIntentClassifier()
        _ml_classifier.load_model("models/intent_classifier")


def detect_intent_ml(formatted_text: str, thread_metadata: Dict[str, Any] = None, 
                    use_lightweight: bool = False) -> str:
    """
    Detect intent using ML models with fallback to rule-based classification.
    
    Args:
        formatted_text: Formatted thread text
        thread_metadata: Optional metadata about the thread
        use_lightweight: Whether to use lightweight model
        
    Returns:
        Detected intent
    """
    global _ml_classifier, _lightweight_classifier
    
    if use_lightweight and _lightweight_classifier:
        intent, _ = _lightweight_classifier.predict_lightweight(formatted_text)
        return intent
    elif _ml_classifier:
        intent, _ = _ml_classifier.predict(formatted_text, thread_metadata)
        return intent
    else:
        # Fallback to rule-based classification
        return classify.detect_intent(formatted_text, thread_metadata)


def get_confidence_score_ml(formatted_text: str, detected_intent: str, 
                           use_lightweight: bool = False) -> float:
    """
    Get confidence score for ML-based classification.
    
    Args:
        formatted_text: Formatted thread text
        detected_intent: The detected intent
        use_lightweight: Whether to use lightweight model
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    global _ml_classifier, _lightweight_classifier
    
    if use_lightweight and _lightweight_classifier:
        _, confidence = _lightweight_classifier.predict_lightweight(formatted_text)
        return confidence
    elif _ml_classifier:
        _, confidence = _ml_classifier.predict(formatted_text)
        return confidence
    else:
        # Fallback to rule-based confidence
        return classify.get_confidence_score(formatted_text, detected_intent) 