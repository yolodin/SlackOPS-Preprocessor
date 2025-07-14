"""
Summarization module for generating summaries of Slack threads.
"""

import re
from typing import List


def generate_summary(formatted_text: str) -> str:
    """
    Generate a summary of the formatted thread text.
    
    For now, this is a basic rule-based approach that will be enhanced later.
    
    Args:
        formatted_text: Formatted string representation of the thread
        
    Returns:
        Generated summary string
    """
    if not formatted_text or not formatted_text.strip():
        return "Empty thread"
    
    lines = formatted_text.strip().split('\n')
    
    # Remove header line
    content_lines = [line for line in lines if not line.startswith('===')]
    
    if not content_lines:
        return "No content found"
    
    # Extract key information
    summary_parts = []
    
    # Count messages and users
    message_count = len(content_lines)
    users = set()
    
    # Extract users and identify potential issues/solutions
    keywords = {
        'error': ['error', 'exception', 'failed', 'broken', 'issue', 'problem'],
        'solution': ['solved', 'fixed', 'resolved', 'working', 'solution', 'thanks'],
        'question': ['how', 'what', 'why', 'when', 'where', '?']
    }
    
    category_counts = {key: 0 for key in keywords}
    
    for line in content_lines:
        # Extract user
        user_match = re.search(r'\] ([^:]+):', line)
        if user_match:
            users.add(user_match.group(1))
        
        # Count keyword categories
        line_lower = line.lower()
        for category, words in keywords.items():
            if any(word in line_lower for word in words):
                category_counts[category] += 1
    
    # Build summary
    summary_parts.append(f"{message_count} messages from {len(users)} users")
    
    # Determine thread type
    if category_counts['error'] > 0:
        if category_counts['solution'] > 0:
            thread_type = "Support issue (resolved)"
        else:
            thread_type = "Support issue (ongoing)"
    elif category_counts['question'] > 0:
        thread_type = "Question/Discussion"
    else:
        thread_type = "General conversation"
    
    summary_parts.append(f"Type: {thread_type}")
    
    # Add first message preview (truncated)
    if content_lines:
        first_message = content_lines[0]
        # Extract just the message text after the timestamp and user
        message_match = re.search(r'\] [^:]+: (.+)', first_message)
        if message_match:
            message_text = message_match.group(1)
            preview = message_text[:100] + "..." if len(message_text) > 100 else message_text
            summary_parts.append(f"Started with: {preview}")
    
    return " | ".join(summary_parts)


def extract_key_phrases(formatted_text: str) -> List[str]:
    """
    Extract key phrases from the formatted text.
    
    Args:
        formatted_text: Formatted string representation of the thread
        
    Returns:
        List of key phrases
    """
    if not formatted_text:
        return []
    
    # Simple keyword extraction
    technical_terms = re.findall(r'\b(?:API|server|database|authentication|deployment|configuration|docker|kubernetes|aws|error|exception|timeout|connection)\b', 
                                formatted_text, re.IGNORECASE)
    
    return list(set(technical_terms)) 