"""
Preprocessing module for formatting Slack thread messages.
"""

from datetime import datetime
from typing import List, Dict, Any


def format_thread_messages(thread_data: Dict[str, Any]) -> str:
    """
    Format Slack thread messages into a readable string format.
    
    Args:
        thread_data: Dictionary containing thread information with messages
        
    Returns:
        Formatted string representation of the thread
    """
    if not thread_data or 'messages' not in thread_data:
        return ""
    
    messages = thread_data['messages']
    formatted_lines = []
    
    # Add thread header
    thread_id = thread_data.get('thread_ts', 'unknown')
    formatted_lines.append(f"=== Thread {thread_id} ===")
    
    for i, message in enumerate(messages):
        # Format timestamp
        timestamp = message.get('ts', '')
        try:
            if timestamp:
                dt = datetime.fromtimestamp(float(timestamp))
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                time_str = "Unknown time"
        except (ValueError, TypeError):
            time_str = "Invalid timestamp"
        
        # Get user and text
        user = message.get('user', 'Unknown User')
        text = message.get('text', '').strip()
        
        # Format message
        formatted_lines.append(f"[{time_str}] {user}: {text}")
    
    return "\n".join(formatted_lines)


def extract_thread_metadata(thread_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from thread data.
    
    Args:
        thread_data: Dictionary containing thread information
        
    Returns:
        Dictionary with extracted metadata
    """
    if not thread_data or 'messages' not in thread_data:
        return {}
    
    messages = thread_data['messages']
    if not messages:
        return {}
    
    # Get first and last message timestamps
    first_ts = messages[0].get('ts')
    last_ts = messages[-1].get('ts')
    
    # Calculate response duration
    duration = 0
    if first_ts and last_ts:
        try:
            duration = float(last_ts) - float(first_ts)
        except (ValueError, TypeError):
            duration = 0
    
    return {
        'thread_id': thread_data.get('thread_ts', 'unknown'),
        'message_count': len(messages),
        'duration_seconds': duration,
        'first_timestamp': first_ts,
        'last_timestamp': last_ts,
        'users': list(set(msg.get('user', 'unknown') for msg in messages))
    } 