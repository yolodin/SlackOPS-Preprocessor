# SlackOPS Preprocessor

This repository contains the AI/NLP backend pipeline for processing Slack support threads from help channels. It's the first stage of a system designed to analyze support interactions, extract insights, and identify automation opportunities in engineering environments.

## Overview

The preprocessor takes raw Slack export data and transforms it into structured, analyzable information. It performs thread formatting, summarization, intent classification, and timing analysis to prepare data for downstream analytics and machine learning workflows.

## Features

- Thread message formatting and preprocessing
- Automatic summarization of conversation threads
- Rule-based intent classification (bug reports, feature requests, questions, etc.)
- Response time calculation and duration analysis
- Robust error handling for production reliability
- Detailed processing statistics and insights

## Project Structure

```
SlackOPS-Preprocessor/
├── run_pipeline.py          # Main execution script
├── preprocess.py            # Message formatting and metadata extraction
├── summarize.py             # Thread summarization logic
├── classify.py              # Intent detection and classification
├── data/
│   └── slack_export_sample.json  # Sample data for testing
└── README.md
```

## Installation

This project requires Python 3.7 or higher. Clone the repository and install dependencies:

```bash
git clone https://github.com/yolodin/SlackOPS-Preprocessor
cd SlackOPS-Preprocessor
# No additional dependencies required - uses Python standard library
```

## Usage

### Basic Usage

Run the pipeline with the sample data:

```bash
python3 run_pipeline.py
```

### Input Format

The pipeline expects JSON data in Slack export format:

```json
[
  {
    "thread_ts": "1671024600.123456",
    "messages": [
      {
        "ts": "1671024600.123456",
        "user": "user1",
        "text": "Message content here"
      }
    ]
  }
]
```

### Output

For each processed thread, the pipeline outputs:

- Thread ID: Unique identifier from Slack
- Summary: Automated description of thread content and type
- Intent: Classified category (bug_report, feature_request, how_to_question, etc.)
- Duration: Time span from first to last message
- Metadata: Message count, user count, confidence scores

## Components

### Preprocessing Module (`preprocess.py`)

Handles raw message formatting and metadata extraction:
- Converts timestamps to human-readable format
- Formats user messages into readable thread representations
- Extracts timing information and participant counts

### Summarization Module (`summarize.py`)

Generates concise summaries of thread content:
- Identifies thread type (support issue, discussion, etc.)
- Extracts key participants and message flow
- Creates preview text from initial messages

### Classification Module (`classify.py`)

Performs rule-based intent detection:
- Bug reports and error discussions
- Feature requests and enhancement proposals
- How-to questions and troubleshooting
- Announcements and general discussions
- Confidence scoring for classification results

## Configuration

The pipeline can be configured by modifying the data file path in `run_pipeline.py`:

```python
data_file = "data/your_slack_export.json"
```

## Error Handling

The pipeline includes comprehensive error handling:
- Individual thread failures don't stop processing
- Detailed error reporting with exception types
- Processing statistics and success rates
- Graceful handling of malformed data

## Development

To extend the pipeline:

1. Add new intent categories: Modify the patterns in `classify.py`
2. Enhance summarization: Update logic in `summarize.py`
3. Custom preprocessing: Extend functions in `preprocess.py`
4. Output formats: Modify result structure in `run_pipeline.py`

## Sample Data

The included sample data contains 5 representative threads:
- Authentication error resolution
- Docker setup question
- Database connection troubleshooting
- Version release announcement
- Feature request discussion

## Performance

Processing time scales approximately linearly with thread count. Typical performance:
- 100 threads: ~2-3 seconds
- 1000 threads: ~20-30 seconds
- Memory usage: <50MB for typical datasets

## Future Enhancements

- Machine learning-based classification models
- Advanced NLP summarization techniques
- Integration with external APIs for enhanced analysis
- Real-time processing capabilities
- Custom output format options
