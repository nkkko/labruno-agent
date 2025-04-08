# Labruno

A Python application that uses Daytona sandboxes and Groq's LLaMA 4 to generate, execute, and evaluate Python code.

## Overview

Labruno is a Flask web application that:

1. Creates a primary sandbox environment
2. Spins up 5 additional sandboxes
3. Presents a simple web interface where users can input a task
4. Uses Groq's LLaMA 4 model to generate Python code in each sandbox based on the user's task
5. Executes the generated code in each sandbox and captures the output
6. Uses LLaMA 4 to evaluate all 5 implementations and determine which is best
7. Presents the results to the user

## Prerequisites

- Python 3.8 or higher
- A Daytona account with API key
- A Groq account with API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/labruno.git
   cd labruno
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Copy the .env.example file to .env and add your API keys:
   ```
   cp .env.example .env
   ```
   Then edit the .env file to add your Daytona and Groq API keys:
   - Get a Daytona API key from the [Daytona Dashboard](https://app.daytona.io/dashboard/)
   - Get a Groq API key from [Groq's website](https://console.groq.com/)

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and go to `http://localhost:5000`

3. Enter a task description in the input box (e.g., "Create a function to calculate the Fibonacci sequence")

4. Click "Generate and Execute Code"

5. Wait for the results (this may take a minute or two as it needs to create multiple sandboxes)

6. View the evaluation results and the generated code from each sandbox

## How It Works

1. The main application creates a primary Daytona sandbox
2. It then creates 5 additional sandboxes
3. When a user submits a task, the task is sent to all 5 sandboxes
4. Each sandbox uses the Groq API to access LLaMA 4 and generate Python code based on the task
5. Each sandbox executes the generated code and captures the output
6. The main sandbox collects all results and uses LLaMA 4 to evaluate which implementation is best
7. All results are returned to the web interface for display

## Notes

- This application requires API keys and will make API calls to both Daytona and Groq services, which may incur costs depending on your usage plans.
- The application will run in test mode with simulated outputs if the API keys are not configured properly.
- You must have Python 3.8+ installed with the packages listed in requirements.txt.

## Recent Fixes

- Fixed an issue where worker sandboxes could not find the task runner script by uploading the script to each sandbox individually.