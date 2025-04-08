# Labruno 🧪🔍

Labruno is an agent coordinator that creates multiple AI solutions to your coding tasks using parallel sandboxes and evaluates them to find the best implementation.

![Labruno Demo Screenshot](https://via.placeholder.com/800x450.png?text=Labruno+Demo)

## What is Labruno?

Labruno acts as an orchestrator that:

1. Takes your coding task and spins up multiple isolated sandboxes
2. Asks each sandbox to generate a unique solution using LLaMA 4
3. Executes all solutions in parallel
4. Uses an LLM as judge to evaluate and select the best implementation
5. Shows you all solutions with the winner highlighted

Think of it as having multiple AI developers working on your task simultaneously, with an expert reviewer choosing the best approach.

## Features at a Glance

- 🏎️ **Parallel Processing**: Creates and runs multiple sandboxes concurrently
- 🧠 **Multiple Solutions**: Generates diverse approaches to the same problem
- 🤖 **AI Evaluation**: Uses an LLM to judge which solution is best
- 🔒 **Secure Execution**: Runs code in isolated Daytona sandboxes
- ⚡ **Fast Results**: Get multiple working implementations in seconds

## Setup

### Requirements
- Python 3.8+
- Daytona and Groq API keys

### Quick Start

```bash
# Install
git clone https://github.com/nkkko/labruno.git
cd labruno
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your API keys to .env

# Run
python app.py
```

### API Keys

In your `.env` file:
```
DAYTONA_API_KEY=your_key_here
DAYTONA_TARGET=us
GROQ_API_KEY=your_key_here
```

## Using Labruno

1. Open http://127.0.0.1:5000 in your browser
2. Type a coding task like "write a function to find prime numbers"
3. Click "Generate and Execute"
4. See multiple solutions and the AI's evaluation of the best one

## How Labruno Works

1. **You provide a task**: Ask for any coding solution
2. **Concurrent sandboxes spin up**: Multiple isolated environments are created in parallel
3. **Each sandbox generates code**: LLaMA 4 creates a unique solution in each environment
4. **All solutions execute**: Code runs safely in isolated sandboxes
5. **AI judges the results**: The LLM evaluates solutions for correctness, efficiency, and style
6. **Results presented**: You see all working solutions with the best one highlighted

## Common Use Cases

- **Interview Prep**: See multiple approaches to coding problems
- **Learning**: Compare different ways to solve the same problem
- **Optimization**: Find the most efficient algorithm for your task
- **Exploration**: Generate diverse implementations and understand trade-offs

## Configuration Options

- Modify number of parallel sandboxes in `app.py`
- Adjust evaluation criteria by changing the ranking in `evaluate_results()`
- Customize prompt templates in `sandbox_task_runner.py`

## Troubleshooting

- **API Key Issues**: Ensure your Daytona and Groq API keys are correctly set in `.env`
- **Slow Results**: For complex tasks, reduce the number of concurrent sandboxes
- **Memory Limitations**: If you encounter memory issues, lower the `max_workers` parameter

## Credits

- [Daytona](https://www.daytona.io/) for sandbox environments
- [Groq](https://groq.com/) for LLaMA 4 access
- [Flask](https://flask.palletsprojects.com/) for the web interface

---

Created by [nkkko](https://github.com/nkkko)