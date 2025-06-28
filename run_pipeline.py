#!/usr/bin/env python3
"""
Main pipeline script for processing Slack thread exports.
"""

import json
import sys
from datetime import datetime
from typing import Dict, Any, List

# Import our custom modules
import preprocess
import summarize
import classify


def load_slack_export(file_path: str) -> List[Dict[str, Any]]:
    """Load Slack export data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Successfully loaded {len(data)} threads from {file_path}")
        return data
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {e}")
        raise


def process_single_thread(thread_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single Slack thread through the entire pipeline."""
    thread_id = thread_data.get('thread_ts', 'unknown')
    
    # Step 1: Format thread messages
    formatted_text = preprocess.format_thread_messages(thread_data)
    
    # Step 2: Extract metadata (including duration calculation)
    metadata = preprocess.extract_thread_metadata(thread_data)
    
    # Step 3: Generate summary
    summary = summarize.generate_summary(formatted_text)
    
    # Step 4: Detect intent
    intent = classify.detect_intent(formatted_text, metadata)
    
    # Step 5: Calculate response duration (already in metadata)
    duration = metadata.get('duration_seconds', 0)
    
    # Format duration in a human-readable way
    if duration > 0:
        if duration < 60:
            duration_str = f"{duration:.1f}s"
        elif duration < 3600:
            duration_str = f"{duration/60:.1f}m"
        else:
            duration_str = f"{duration/3600:.1f}h"
    else:
        duration_str = "0s"
    
    return {
        'thread_id': thread_id,
        'summary': summary,
        'intent': intent,
        'duration': duration_str,
        'duration_seconds': duration,
        'message_count': metadata.get('message_count', 0),
        'user_count': len(metadata.get('users', [])),
        'confidence': classify.get_confidence_score(formatted_text, intent)
    }


def main():
    """Main function that orchestrates the entire pipeline."""
    # Configuration
    data_file = "data/slack_export_sample.json"
    
    print("=" * 60)
    print("SLACK THREAD ANALYZER - PREPROCESSING PIPELINE")
    print("=" * 60)
    print()
    
    try:
        # Load the data
        threads = load_slack_export(data_file)
        
        if not threads:
            print("No threads found in the data file.")
            return
        
        print(f"\nProcessing {len(threads)} threads...")
        print("-" * 60)
        
        successful_processes = 0
        failed_processes = 0
        results = []
        
        # Process each thread
        for i, thread_data in enumerate(threads, 1):
            thread_id = thread_data.get('thread_ts', f'thread_{i}')
            
            try:
                print(f"\nProcessing Thread {i}/{len(threads)} (ID: {thread_id})")
                
                # Process the thread
                result = process_single_thread(thread_data)
                results.append(result)
                
                # Print the result
                print("✓ Successfully processed:")
                print(f"  Thread ID: {result['thread_id']}")
                print(f"  Summary: {result['summary']}")
                print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
                print(f"  Duration: {result['duration']}")
                print(f"  Messages: {result['message_count']}, Users: {result['user_count']}")
                
                successful_processes += 1
                
            except Exception as e:
                print(f"✗ Failed to process thread {thread_id}: {str(e)}")
                print(f"  Error type: {type(e).__name__}")
                failed_processes += 1
                
                # Add a failed result entry
                results.append({
                    'thread_id': thread_id,
                    'summary': 'Processing failed',
                    'intent': 'error',
                    'duration': '0s',
                    'error': str(e)
                })
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total threads: {len(threads)}")
        print(f"Successfully processed: {successful_processes}")
        print(f"Failed: {failed_processes}")
        print(f"Success rate: {(successful_processes/len(threads)*100):.1f}%")
        
        # Print aggregated insights
        if successful_processes > 0:
            print("\n" + "-" * 60)
            print("INSIGHTS")
            print("-" * 60)
            
            # Intent distribution
            intent_counts = {}
            total_duration = 0
            total_messages = 0
            
            for result in results:
                if 'error' not in result:
                    intent = result['intent']
                    intent_counts[intent] = intent_counts.get(intent, 0) + 1
                    total_duration += result.get('duration_seconds', 0)
                    total_messages += result.get('message_count', 0)
            
            print("Intent Distribution:")
            for intent, count in sorted(intent_counts.items()):
                percentage = (count / successful_processes) * 100
                print(f"  {intent}: {count} ({percentage:.1f}%)")
            
            if successful_processes > 0:
                avg_duration = total_duration / successful_processes
                avg_messages = total_messages / successful_processes
                print(f"\nAverage thread duration: {avg_duration:.1f}s")
                print(f"Average messages per thread: {avg_messages:.1f}")
        
        print("\n" + "=" * 60)
        print("All results have been processed and displayed above.")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        sys.exit(1)


if __name__ == "__main__":
    main()
