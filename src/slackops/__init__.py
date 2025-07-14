"""
SlackOPS-Preprocessor: A comprehensive toolkit for processing Slack data exports.

This package provides tools for:
- Data standardization and validation
- Machine learning-based classification and summarization
- Analytics and reporting
"""

__version__ = "1.0.0"
__author__ = "SlackOPS Team"

# Import key classes and functions for easy access
from .slack_data_adapter import SlackDataAdapter, create_adapter_config, quick_setup
from .classify_ml import MLIntentClassifier, SentenceEmbeddingClassifier
from .summarize_ml import AbstractiveSummarizer, ExtractiveSummarizer, HybridSummarizer

__all__ = [
    "SlackDataAdapter",
    "create_adapter_config", 
    "quick_setup",
    "MLIntentClassifier",
    "SentenceEmbeddingClassifier",
    "AbstractiveSummarizer",
    "ExtractiveSummarizer", 
    "HybridSummarizer"
] 