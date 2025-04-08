# Labruno 🧪🔍

A Python application that leverages Daytona sandboxes and Groq's LLaMA 4 model to generate, execute, and evaluate multiple implementations of Python code from a single prompt.

![Labruno Demo Screenshot](https://via.placeholder.com/800x450.png?text=Labruno+Demo)

## Overview

Labruno is an AI-powered code sandbox platform that:

1. Spins up multiple isolated Daytona sandbox environments concurrently
2. Uses LLaMA 4 to generate different Python implementations of the same task
3. Executes each implementation securely in isolated sandboxes
4. Evaluates all implementations to determine the optimal solution
5. Presents the results through a clean, intuitive web interface

Think of it as parallel AI coding with immediate execution and evaluation - letting you quickly explore different coding approaches to the same problem.

## Key Features

- 🧠 **Multiple AI Solutions**: Generates 5 unique implementations for each task
- 🔒 **Secure Execution**: Runs all code in isolated Daytona sandboxes
- 🔄 **Parallel Processing**: Handles code generation and evaluation concurrently
- 📊 **Implementation Comparison**: Analyzes which solution is most effective
- 🖥️ **Clean Web Interface**: Simple input/output for quick experimentation

## Prerequisites

- Python 3.8+
- [Daytona account](https://app.daytona.io/) with API key
- [Groq account](https://console.groq.com/) with API key
- pip package manager

## Quick Installation

```bash
# Clone the repository
git clone https://github.com/nkkko/labruno-agent.git
cd labruno-agent

# Set up virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Edit your `.env` file with the required API keys:

```
DAYTONA_API_KEY=your_daytona_api_key_here
DAYTONA_TARGET=us
GROQ_API_KEY=your_groq_api_key_here
```

- **DAYTONA_API_KEY**: Get from the [Daytona Dashboard](https://app.daytona.io/dashboard/) → API Keys → Create Key
- **DAYTONA_TARGET**: Geographic location for sandbox creation (usually 'us')
- **GROQ_API_KEY**: Get from [Groq Console](https://console.groq.com/) → API Keys

## Running Labruno

1. Start the application:
   ```bash
   python app.py
   ```

2. Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. Enter a coding task like:
   - "Sort a list of dictionaries by a value"
   - "Create a function that calculates prime numbers"
   - "Find the longest palindrome in a string"

4. Click "Generate and Execute Code" and wait for your results (typically 30-60 seconds)

## Example Use Cases

- **Algorithm Exploration**: Generate multiple approaches to the same algorithm
- **Code Optimization**: Find more efficient ways to solve a problem
- **Learning Tool**: See different ways to implement the same functionality
- **Interview Prep**: Practice explaining the tradeoffs between implementations

## How It Works

![Labruno Architecture](https://via.placeholder.com/800x400.png?text=Labruno+Architecture)

1. **Request Processing**: User submits a coding task through the web interface
2. **Sandbox Creation**: Application creates a main sandbox and 5 worker sandboxes
3. **Code Generation**: Each sandbox uses LLaMA 4 to generate unique Python solutions
4. **Secure Execution**: Code is executed in isolated environments with results captured
5. **Evaluation**: All implementations are compared for correctness, efficiency, and style
6. **Result Presentation**: The best implementation is highlighted and all results displayed

## Troubleshooting

If you encounter issues:

- **Empty Results**: Check that your API keys are correctly configured in `.env`
- **Connection Errors**: Verify your internet connection and Daytona/Groq API status
- **Timeout Errors**: For complex tasks, try simplifying your request

## Advanced Configuration

- Modify `app.py` to adjust number of parallel sandboxes
- Edit prompt templates in `sandbox_task_runner.py` to customize code generation
- Change timeout settings for long-running code execution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Daytona](https://www.daytona.io/) for providing the sandbox environments
- [Groq](https://groq.com/) for access to the LLaMA 4 model
- [Flask](https://flask.palletsprojects.com/) for the web framework

---

Built with ❤️ and 🧪 by [nkkko](https://github.com/nkkko)