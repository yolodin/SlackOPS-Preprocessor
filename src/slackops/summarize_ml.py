"""
ML-based summarization module for generating summaries of Slack threads.
"""

import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, pipeline,
    BartTokenizer, BartForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import torch

# Fallback to rule-based summarization
from . import summarize


class AbstractiveSummarizer:
    """
    Abstractive summarization using transformer models (BART, T5, etc.)
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the abstractive summarizer.
        
        Args:
            model_name: Hugging Face model name for summarization
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.max_length = 512
        self.min_length = 50
        
    def load_model(self):
        """Load the summarization model."""
        try:
            print(f"Loading summarization model: {self.model_name}")
            
            # Initialize model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Create pipeline
            self.pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                framework="pt"
            )
            
            print("Summarization model loaded successfully")
            
        except Exception as e:
            print(f"Error loading summarization model: {e}")
            self.pipeline = None
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
        """
        Generate an abstractive summary of the input text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
            
        Returns:
            Generated summary
        """
        if not self.pipeline:
            # Fallback to rule-based summarization
            return summarize.generate_summary(text)
        
        try:
            # Preprocess text
            processed_text = self._preprocess_for_summarization(text)
            
            if len(processed_text.split()) < 10:
                return "Text too short for summarization"
            
            # Generate summary
            summary_result = self.pipeline(
                processed_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            
            return summary_result[0]['summary_text']
            
        except Exception as e:
            print(f"Error in abstractive summarization: {e}")
            # Fallback to rule-based summarization
            return summarize.generate_summary(text)
    
    def _preprocess_for_summarization(self, text: str) -> str:
        """
        Preprocess text for better summarization.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove thread header and metadata
        lines = text.split('\n')
        content_lines = []
        
        for line in lines:
            # Skip header lines
            if line.startswith('===') or line.startswith('---'):
                continue
                
            # Extract message content (remove timestamps and usernames)
            if ']:' in line:
                # Extract text after username
                parts = line.split(']: ', 1)
                if len(parts) == 2:
                    content_lines.append(parts[1])
            else:
                content_lines.append(line)
        
        # Join and clean
        processed_text = ' '.join(content_lines)
        
        # Remove extra whitespace
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        return processed_text


class ExtractiveSummarizer:
    """
    Extractive summarization using sentence ranking and selection.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the extractive summarizer.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.sentence_model = SentenceTransformer(model_name)
        self.stemmer = PorterStemmer()
        
        # Download NLTK data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def summarize_extractive(self, text: str, num_sentences: int = 3) -> str:
        """
        Generate an extractive summary by selecting key sentences.
        
        Args:
            text: Input text to summarize
            num_sentences: Number of sentences to extract
            
        Returns:
            Extractive summary
        """
        try:
            # Preprocess text
            processed_text = self._preprocess_for_extraction(text)
            
            if not processed_text:
                return "No content to summarize"
            
            # Split into sentences
            sentences = sent_tokenize(processed_text)
            
            if len(sentences) <= num_sentences:
                return ' '.join(sentences)
            
            # Rank sentences
            ranked_sentences = self._rank_sentences(sentences)
            
            # Select top sentences
            top_sentences = ranked_sentences[:num_sentences]
            
            # Sort by original order
            top_sentences.sort(key=lambda x: x[1])
            
            # Extract text
            summary_sentences = [sent[0] for sent in top_sentences]
            
            return ' '.join(summary_sentences)
            
        except Exception as e:
            print(f"Error in extractive summarization: {e}")
            # Fallback to rule-based summarization
            return summarize.generate_summary(text)
    
    def _preprocess_for_extraction(self, text: str) -> str:
        """
        Preprocess text for extractive summarization.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Remove thread header and metadata
        lines = text.split('\n')
        content_lines = []
        
        for line in lines:
            # Skip header lines
            if line.startswith('===') or line.startswith('---'):
                continue
                
            # Extract message content
            if ']:' in line:
                parts = line.split(']: ', 1)
                if len(parts) == 2:
                    content_lines.append(parts[1])
            else:
                content_lines.append(line)
        
        return ' '.join(content_lines)
    
    def _rank_sentences(self, sentences: List[str]) -> List[tuple]:
        """
        Rank sentences by importance using multiple criteria.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of (sentence, original_index, score) tuples sorted by score
        """
        # Generate embeddings for all sentences
        embeddings = self.sentence_model.encode(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = np.inner(embeddings, embeddings)
        
        # Calculate scores for each sentence
        sentence_scores = []
        
        for i, sentence in enumerate(sentences):
            # Position score (earlier sentences are more important)
            position_score = 1.0 / (i + 1)
            
            # Length score (prefer medium-length sentences)
            length_score = self._calculate_length_score(sentence)
            
            # Keyword score (technical terms, question words, etc.)
            keyword_score = self._calculate_keyword_score(sentence)
            
            # Centrality score (similarity to other sentences)
            centrality_score = np.mean(similarity_matrix[i])
            
            # Combine scores
            total_score = (
                0.3 * position_score +
                0.2 * length_score +
                0.3 * keyword_score +
                0.2 * centrality_score
            )
            
            sentence_scores.append((sentence, i, total_score))
        
        # Sort by score (descending)
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        
        return sentence_scores
    
    def _calculate_length_score(self, sentence: str) -> float:
        """
        Calculate length score for a sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Length score between 0 and 1
        """
        words = word_tokenize(sentence.lower())
        word_count = len(words)
        
        # Prefer sentences with 10-30 words
        if word_count < 5:
            return 0.2
        elif word_count < 10:
            return 0.6
        elif word_count <= 30:
            return 1.0
        else:
            return 0.7
    
    def _calculate_keyword_score(self, sentence: str) -> float:
        """
        Calculate keyword importance score.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Keyword score between 0 and 1
        """
        sentence_lower = sentence.lower()
        
        # Technical keywords
        technical_keywords = [
            'error', 'exception', 'bug', 'issue', 'problem', 'solution',
            'api', 'database', 'server', 'authentication', 'configuration',
            'deployment', 'docker', 'kubernetes', 'aws', 'feature', 'request'
        ]
        
        # Question/action keywords
        action_keywords = [
            'how', 'what', 'why', 'when', 'where', 'fixed', 'resolved',
            'working', 'solved', 'help', 'need', 'want', 'would', 'should'
        ]
        
        # Count keyword matches
        tech_matches = sum(1 for kw in technical_keywords if kw in sentence_lower)
        action_matches = sum(1 for kw in action_keywords if kw in sentence_lower)
        
        # Normalize scores
        tech_score = min(1.0, tech_matches / 3)
        action_score = min(1.0, action_matches / 2)
        
        return 0.6 * tech_score + 0.4 * action_score


class HybridSummarizer:
    """
    Hybrid summarizer combining extractive and abstractive approaches.
    """
    
    def __init__(self, abstractive_model: str = "facebook/bart-large-cnn",
                 extractive_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the hybrid summarizer.
        
        Args:
            abstractive_model: Model for abstractive summarization
            extractive_model: Model for extractive summarization
        """
        self.abstractive_summarizer = AbstractiveSummarizer(abstractive_model)
        self.extractive_summarizer = ExtractiveSummarizer(extractive_model)
        self.models_loaded = False
    
    def load_models(self):
        """Load both summarization models."""
        try:
            self.abstractive_summarizer.load_model()
            self.models_loaded = True
            print("Hybrid summarizer models loaded successfully")
        except Exception as e:
            print(f"Error loading hybrid summarizer: {e}")
            self.models_loaded = False
    
    def generate_summary(self, text: str, approach: str = "hybrid") -> str:
        """
        Generate summary using specified approach.
        
        Args:
            text: Input text to summarize
            approach: 'extractive', 'abstractive', or 'hybrid'
            
        Returns:
            Generated summary
        """
        if not text or not text.strip():
            return "Empty thread"
        
        if approach == "extractive":
            return self.extractive_summarizer.summarize_extractive(text)
        elif approach == "abstractive":
            if self.models_loaded:
                return self.abstractive_summarizer.summarize_text(text)
            else:
                # Fallback to extractive
                return self.extractive_summarizer.summarize_extractive(text)
        else:  # hybrid
            return self._generate_hybrid_summary(text)
    
    def _generate_hybrid_summary(self, text: str) -> str:
        """
        Generate a hybrid summary combining both approaches.
        
        Args:
            text: Input text
            
        Returns:
            Hybrid summary
        """
        try:
            # First, get key sentences using extractive method
            key_sentences = self.extractive_summarizer.summarize_extractive(text, num_sentences=5)
            
            # Then, generate abstractive summary from key sentences
            if self.models_loaded and len(key_sentences.split()) > 20:
                abstractive_summary = self.abstractive_summarizer.summarize_text(
                    key_sentences, max_length=100, min_length=20
                )
                
                # Combine both summaries
                return f"{abstractive_summary.strip()}"
            else:
                # Use extractive summary if abstractive model not available
                return key_sentences
                
        except Exception as e:
            print(f"Error in hybrid summarization: {e}")
            # Fallback to rule-based summarization
            return summarize.generate_summary(text)


# Global summarizer instance
_hybrid_summarizer = None


def initialize_ml_summarizer(abstractive_model: str = "facebook/bart-large-cnn",
                            extractive_model: str = "all-MiniLM-L6-v2"):
    """
    Initialize ML summarization models.
    
    Args:
        abstractive_model: Model for abstractive summarization
        extractive_model: Model for extractive summarization
    """
    global _hybrid_summarizer
    
    _hybrid_summarizer = HybridSummarizer(abstractive_model, extractive_model)
    _hybrid_summarizer.load_models()


def generate_summary_ml(formatted_text: str, approach: str = "hybrid") -> str:
    """
    Generate summary using ML models with fallback to rule-based summarization.
    
    Args:
        formatted_text: Formatted thread text
        approach: 'extractive', 'abstractive', or 'hybrid'
        
    Returns:
        Generated summary
    """
    global _hybrid_summarizer
    
    if _hybrid_summarizer:
        return _hybrid_summarizer.generate_summary(formatted_text, approach)
    else:
        # Fallback to rule-based summarization
        return summarize.generate_summary(formatted_text)


def extract_key_insights(formatted_text: str) -> Dict[str, Any]:
    """
    Extract key insights from thread using NLP techniques.
    
    Args:
        formatted_text: Formatted thread text
        
    Returns:
        Dictionary containing insights
    """
    try:
        # Initialize components if not already done
        if not _hybrid_summarizer:
            initialize_ml_summarizer()
        
        # Generate different types of summaries
        extractive_summary = generate_summary_ml(formatted_text, "extractive")
        
        # Extract entities and topics
        entities = _extract_entities(formatted_text)
        topics = _extract_topics(formatted_text)
        
        # Analyze sentiment and urgency
        sentiment = _analyze_sentiment(formatted_text)
        urgency = _analyze_urgency(formatted_text)
        
        return {
            "extractive_summary": extractive_summary,
            "entities": entities,
            "topics": topics,
            "sentiment": sentiment,
            "urgency": urgency
        }
        
    except Exception as e:
        print(f"Error extracting insights: {e}")
        return {
            "extractive_summary": summarize.generate_summary(formatted_text),
            "entities": [],
            "topics": [],
            "sentiment": "neutral",
            "urgency": "medium"
        }


def _extract_entities(text: str) -> List[str]:
    """Extract named entities from text."""
    # Simple regex-based entity extraction
    entities = []
    
    # Technical terms
    tech_pattern = r'\b(?:API|REST|JSON|XML|SQL|Docker|Kubernetes|AWS|GCP|Azure|Redis|MongoDB|PostgreSQL|MySQL)\b'
    tech_entities = re.findall(tech_pattern, text, re.IGNORECASE)
    entities.extend(tech_entities)
    
    # Error codes
    error_pattern = r'\b(?:error|exception|status)\s*:?\s*\d+\b'
    error_entities = re.findall(error_pattern, text, re.IGNORECASE)
    entities.extend(error_entities)
    
    return list(set(entities))


def _extract_topics(text: str) -> List[str]:
    """Extract main topics from text."""
    topics = []
    
    topic_keywords = {
        'authentication': ['auth', 'login', 'token', 'credential', 'permission'],
        'deployment': ['deploy', 'release', 'build', 'ci/cd', 'pipeline'],
        'performance': ['slow', 'timeout', 'latency', 'performance', 'optimization'],
        'database': ['database', 'db', 'query', 'connection', 'sql'],
        'api': ['api', 'endpoint', 'request', 'response', 'integration'],
        'configuration': ['config', 'setting', 'environment', 'parameter']
    }
    
    text_lower = text.lower()
    for topic, keywords in topic_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            topics.append(topic)
    
    return topics


def _analyze_sentiment(text: str) -> str:
    """Analyze sentiment of the text."""
    # Simple rule-based sentiment analysis
    text_lower = text.lower()
    
    positive_words = ['solved', 'fixed', 'working', 'thanks', 'great', 'perfect', 'awesome']
    negative_words = ['error', 'failed', 'broken', 'issue', 'problem', 'stuck', 'frustrated']
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"


def _analyze_urgency(text: str) -> str:
    """Analyze urgency level of the text."""
    text_lower = text.lower()
    
    urgent_words = ['urgent', 'critical', 'asap', 'immediately', 'emergency', 'production', 'down']
    high_words = ['soon', 'quickly', 'fast', 'priority', 'important']
    
    urgent_count = sum(1 for word in urgent_words if word in text_lower)
    high_count = sum(1 for word in high_words if word in text_lower)
    
    if urgent_count > 0:
        return "urgent"
    elif high_count > 0:
        return "high"
    else:
        return "medium" 