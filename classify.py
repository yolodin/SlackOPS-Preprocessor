"""
Classification module for detecting intent in Slack threads.
"""

import re
from typing import Dict, List, Any


def detect_intent(formatted_text: str, thread_metadata: Dict[str, Any] = None) -> str:
    """
    Detect the intent/category of a Slack thread using rule-based classification.
    
    Args:
        formatted_text: Formatted string representation of the thread
        thread_metadata: Optional metadata about the thread
        
    Returns:
        Detected intent/category label
    """
    if not formatted_text or not formatted_text.strip():
        return "unknown"
    
    text_lower = formatted_text.lower()
    
    # Define intent patterns
    intent_patterns = {
        'bug_report': [
            r'\berror\b', r'\bexception\b', r'\bfailed\b', r'\bbroken\b', 
            r'\bbug\b', r'\bissue\b', r'\bcrash\b', r'\bnot working\b'
        ],
        'feature_request': [
            r'\bfeature\b', r'\brequest\b', r'\benhancement\b', r'\bwould like\b',
            r'\bcan we\b', r'\bwish\b', r'\bproposal\b'
        ],
        'how_to_question': [
            r'\bhow to\b', r'\bhow do i\b', r'\bhow can i\b', r'\bsteps\b',
            r'\btutorial\b', r'\bguide\b', r'\binstructions\b'
        ],
        'troubleshooting': [
            r'\btrouble\b', r'\bproblem\b', r'\bhelp\b', r'\bstuck\b',
            r'\bdebug\b', r'\bfixing\b', r'\bresolve\b'
        ],
        'configuration': [
            r'\bconfig\b', r'\bsetup\b', r'\binstall\b', r'\benvironment\b',
            r'\bsettings\b', r'\bparameters\b', r'\bdeployment\b'
        ],
        'discussion': [
            r'\bthoughts\b', r'\bopinion\b', r'\bdiscuss\b', r'\bwhat do you think\b',
            r'\bfeedback\b', r'\bsuggestion\b', r'\bidea\b'
        ],
        'announcement': [
            r'\bannounce\b', r'\brelease\b', r'\bupdate\b', r'\bnew version\b',
            r'\bfyi\b', r'\bheads up\b', r'\bnotice\b'
        ]
    }
    
    # Score each intent
    intent_scores = {}
    
    for intent, patterns in intent_patterns.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower))
            score += matches
        intent_scores[intent] = score
    
    # Find the highest scoring intent
    if intent_scores and max(intent_scores.values()) > 0:
        detected_intent = max(intent_scores, key=intent_scores.get)
    else:
        detected_intent = "general"
    
    # Apply some additional rules based on metadata
    if thread_metadata:
        message_count = thread_metadata.get('message_count', 0)
        duration = thread_metadata.get('duration_seconds', 0)
        
        # Long threads with many messages might be discussions
        if message_count > 10 and duration > 3600:  # More than 1 hour
            if detected_intent == "general":
                detected_intent = "discussion"
        
        # Very short threads might be announcements
        elif message_count <= 2 and detected_intent == "general":
            detected_intent = "announcement"
    
    return detected_intent


def get_confidence_score(formatted_text: str, detected_intent: str) -> float:
    """
    Calculate confidence score for the detected intent.
    
    Args:
        formatted_text: Formatted string representation of the thread
        detected_intent: The detected intent
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not formatted_text or detected_intent == "unknown":
        return 0.0
    
    # Simple confidence calculation based on keyword matches
    text_lower = formatted_text.lower()
    
    intent_keywords = {
        'bug_report': ['error', 'exception', 'failed', 'broken', 'bug', 'issue'],
        'feature_request': ['feature', 'request', 'enhancement', 'would like'],
        'how_to_question': ['how to', 'how do', 'steps', 'tutorial'],
        'troubleshooting': ['trouble', 'problem', 'help', 'stuck'],
        'configuration': ['config', 'setup', 'install', 'environment'],
        'discussion': ['thoughts', 'opinion', 'discuss', 'feedback'],
        'announcement': ['announce', 'release', 'update', 'fyi']
    }
    
    if detected_intent in intent_keywords:
        keywords = intent_keywords[detected_intent]
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        # Normalize by number of keywords and thread length
        word_count = len(text_lower.split())
        confidence = min(1.0, (matches * 10) / max(word_count, 10))
        return confidence
    
    return 0.5  # Default confidence for general intent 