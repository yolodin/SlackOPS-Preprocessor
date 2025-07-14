#!/usr/bin/env python3
"""
Demo script showing how to use the Slack Data Adapter
with the existing ML pipeline without modifying any working code.
"""

import os
import sys
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from slackops.slack_data_adapter import SlackDataAdapter, quick_setup


def demo_adapter_with_existing_pipeline():
    """
    Demonstrate how the adapter integrates with the existing pipeline.
    """
    print("=" * 80)
    print("SLACK DATA ADAPTER DEMO")
    print("=" * 80)
    
    # 1. Quick setup - detect and convert all Slack data
    print("\n1. QUICK SETUP - Converting Slack data...")
    standardized_file = quick_setup("data/")
    
    if not standardized_file:
        print("No data to process. Exiting.")
        return
    
    # 2. Show how to use with existing ML pipeline
    print("\n2. INTEGRATING WITH ML PIPELINE...")
    print("Now you can use the converted data with any of these commands:")
    print()
    
    # Rule-based processing
    print("🔹 Rule-based processing:")
    print(f"python3 run_pipeline.py --data-file {standardized_file}")
    print()
    
    # ML-based processing
    print("🔹 ML-based processing (lightweight):")
    print(f"python3 run_pipeline_ml.py --use-ml --lightweight --data-file {standardized_file}")
    print()
    
    # Comparison mode
    print("🔹 Compare rule-based vs ML approaches:")
    print(f"python3 run_pipeline_ml.py --compare --data-file {standardized_file}")
    print()
    
    # Training on your data
    print("🔹 Train ML models on your data:")
    print(f"python3 train_models.py --data-file {standardized_file}")
    print()
    
    # 3. Show adapter capabilities
    print("\n3. ADAPTER CAPABILITIES...")
    
    adapter = SlackDataAdapter("data/")
    
    # Load and analyze the converted data
    with open(standardized_file, 'r') as f:
        threads = json.load(f)
    
    # Show validation report
    validation_report = adapter.validate_data(threads)
    
    print(f"📊 Data Analysis:")
    print(f"   • Total threads: {validation_report['total_threads']}")
    print(f"   • Total messages: {validation_report['total_messages']}")
    print(f"   • Unique users: {validation_report['unique_users']}")
    
    if validation_report['date_range']['earliest']:
        print(f"   • Date range: {validation_report['date_range']['earliest'].strftime('%Y-%m-%d')} to {validation_report['date_range']['latest'].strftime('%Y-%m-%d')}")
    
    if validation_report['issues']:
        print(f"   • Data issues: {len(validation_report['issues'])}")
    else:
        print("   • Data quality: ✅ No issues found")
    
    print("\n4. RUNNING LIVE DEMO...")
    
    # Actually run the ML pipeline on the converted data
    return standardized_file


def demo_format_detection():
    """
    Demonstrate automatic format detection capabilities.
    """
    print("\n" + "=" * 80)
    print("FORMAT DETECTION DEMO")
    print("=" * 80)
    
    adapter = SlackDataAdapter("data/")
    
    # Find all JSON files
    import glob
    json_files = glob.glob("data/*.json")
    
    print(f"Found {len(json_files)} JSON files in data/ directory:")
    print()
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        detected_format = adapter.detect_format(file_path)
        
        print(f"📄 {filename}")
        print(f"   Format: {detected_format}")
        
        # Show a preview of the data structure
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if data and isinstance(data, list):
                sample = data[0]
                print(f"   Preview: {list(sample.keys())[:5]}...")
                print(f"   Records: {len(data)}")
        except:
            print("   Error reading file")
        
        print()


def demo_custom_format_handling():
    """
    Show how to handle custom Slack export formats.
    """
    print("\n" + "=" * 80)
    print("CUSTOM FORMAT HANDLING DEMO")
    print("=" * 80)
    
    # Create a sample custom format file
    custom_data = [
        {
            "conversation_id": "conv_123",
            "channel_name": "support",
            "posts": [
                {
                    "author": "alice",
                    "created_at": "2024-01-15T10:30:00Z",
                    "content": "Need help with API authentication"
                },
                {
                    "author": "bob",
                    "created_at": "2024-01-15T10:32:00Z",
                    "content": "Check your API key configuration"
                }
            ]
        }
    ]
    
    # Save custom format
    custom_file = "data/custom_format_example.json"
    with open(custom_file, 'w') as f:
        json.dump(custom_data, f, indent=2)
    
    print(f"📝 Created custom format example: {custom_file}")
    
    # Show how the adapter handles it
    adapter = SlackDataAdapter("data/")
    
    # The adapter will try to convert even unusual formats
    print("\n🔄 Converting custom format...")
    try:
        # Load with custom field mapping
        with open(custom_file, 'r') as f:
            raw_data = json.load(f)
        
        # Manually adapt the unusual format
        adapted_data = []
        for conv in raw_data:
            thread = {
                "thread_ts": conv["conversation_id"],
                "channel": conv.get("channel_name", "unknown"),
                "messages": []
            }
            
            for post in conv.get("posts", []):
                message = {
                    "timestamp": post["created_at"],
                    "user": post["author"],
                    "text": post["content"]
                }
                thread["messages"].append(message)
            
            adapted_data.append(thread)
        
        # Now convert using the adapter
        standardized = adapter.convert_custom_format(adapted_data)
        
        print(f"✅ Successfully converted {len(standardized)} threads")
        print("   Standardized format ready for ML pipeline!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Clean up
    if os.path.exists(custom_file):
        os.remove(custom_file)


def run_live_pipeline_demo(standardized_file: str):
    """
    Actually run the ML pipeline on the converted data.
    """
    print("\n" + "=" * 80)
    print("LIVE PIPELINE DEMO")
    print("=" * 80)
    
    print("🚀 Running ML pipeline on your converted Slack data...")
    
    # Import the existing pipeline (no modifications needed!)
    try:
        import run_pipeline_ml
        import sys
        
        # Temporarily modify sys.argv to pass arguments
        original_argv = sys.argv.copy()
        sys.argv = [
            'run_pipeline_ml.py',
            '--use-ml',
            '--lightweight',
            '--max-threads', '3',
            '--data-file', standardized_file
        ]
        
        # Run the existing pipeline
        run_pipeline_ml.main()
        
        # Restore original argv
        sys.argv = original_argv
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        print("You can manually run:")
        print(f"python3 run_pipeline_ml.py --use-ml --lightweight --data-file {standardized_file}")


def main():
    """
    Main demo function.
    """
    print("🎯 SLACK DATA ADAPTER INTEGRATION DEMO")
    print("=" * 80)
    print()
    print("This demo shows how to integrate your Slack data with the")
    print("existing ML pipeline WITHOUT modifying any working code!")
    print()
    
    # Run format detection demo first
    demo_format_detection()
    
    # Run the main adapter demo
    standardized_file = demo_adapter_with_existing_pipeline()
    
    # Show custom format handling
    demo_custom_format_handling()
    
    # Run live demo if we have data
    if standardized_file and os.path.exists(standardized_file):
        run_live_pipeline_demo(standardized_file)
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE! 🎉")
    print("=" * 80)
    print()
    print("Key Benefits of the Adapter Approach:")
    print("✅ No modifications to existing working code")
    print("✅ Handles multiple Slack export formats")
    print("✅ Automatic format detection")
    print("✅ Data validation and quality reporting")
    print("✅ Seamless integration with ML pipeline")
    print()
    print("Next Steps:")
    print("1. Add your Slack export files to the data/ directory")
    print("2. Run: python3 slack_data_adapter.py --setup")
    print("3. Use the generated file with the ML pipeline")


if __name__ == "__main__":
    main() 