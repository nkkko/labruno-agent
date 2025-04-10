import os
import json
import sys
from groq import Groq

# Get API key from environment (will be set by process.exec)
print("DEBUG[sandbox]: GROQ_API_KEY environment variable is present:", bool(os.environ.get("GROQ_API_KEY")))

def generate_code(user_input):
    """Generate Python code based on user input using Groq"""
    import time

    # Initialize Groq client
    groq_api_key = os.environ.get("GROQ_API_KEY")
    print(f"DEBUG[sandbox]: Using GROQ_API_KEY: {'Present' if groq_api_key else 'Missing'}")
    client = Groq(api_key=groq_api_key)

    # Get model from environment
    model = os.environ.get("GROQ_MODEL")
    print(f"DEBUG[sandbox]: Using model: {model}")

    # Detailed timing metrics
    timing = {
        'client_init_start': time.time(),
        'api_prep_start': 0,
        'api_call_start': 0,
        'api_call_end': 0,
        'processing_end': 0
    }

    # Log API request preparation start
    timing['api_prep_start'] = time.time()

    # User input information
    input_info = {
        'length': len(user_input),
        'token_estimate': len(user_input.split()) # rough estimate
    }

    # Log API request
    request_time = time.time()
    print(f"API_REQUEST_LOG: model={model}, time={request_time}, user_input_length={input_info['length']}, token_estimate={input_info['token_estimate']}")

    # Start timing the API call specifically
    timing['api_call_start'] = time.time()

    try:
        # Generate code using model with a more specific prompt
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that generates Python code based on user requests. Always provide your code inside ```python code blocks. Even if you want to provide multiple code examples, include each inside its own ```python code block. Our code executor will only extract and run the code from within the first python code block."
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            model=model,
        )

        # Record API call end time
        timing['api_call_end'] = time.time()

        # Calculate API call duration
        api_call_duration = timing['api_call_end'] - timing['api_call_start']

        # Log API response with detailed timing
        response_time = time.time()

        # Get token info if available - handle different API response formats
        try:
            # Try to get usage information safely
            usage = getattr(chat_completion, 'usage', None)
            
            # Different models/API versions might have different attribute structures
            if usage:
                if hasattr(usage, 'prompt_tokens'):
                    # Direct attribute access
                    input_tokens = usage.prompt_tokens
                    output_tokens = getattr(usage, 'completion_tokens', 0)
                elif hasattr(usage, '__getitem__'):
                    # Dictionary-like access
                    input_tokens = usage.get('prompt_tokens', 0)
                    output_tokens = usage.get('completion_tokens', 0)
                else:
                    # Fall back to default values
                    input_tokens = 0
                    output_tokens = 0
            else:
                input_tokens = 0
                output_tokens = 0
                
            print(f"API_RESPONSE_LOG: model={model}, time={response_time}, " +
                  f"duration={api_call_duration:.2f}s, status=success, " +
                  f"input_tokens={input_tokens}, output_tokens={output_tokens}")
        except Exception as token_err:
            # If anything goes wrong with token counting, log it but continue
            print(f"API_RESPONSE_LOG: model={model}, time={response_time}, " +
                  f"duration={api_call_duration:.2f}s, status=success, " +
                  f"token_error={str(token_err)}")

        # Process the response
        generated_code = chat_completion.choices[0].message.content

        # Record end of processing
        timing['processing_end'] = time.time()

        # Calculate full duration (total time)
        total_duration = timing['processing_end'] - timing['client_init_start']

        # Calculate time spent in each phase
        timing_breakdown = {
            'preparation': timing['api_call_start'] - timing['api_prep_start'],
            'api_call': timing['api_call_end'] - timing['api_call_start'],
            'processing': timing['processing_end'] - timing['api_call_end'],
            'total': total_duration
        }

        print(f"DEBUG[sandbox]: GROQ API timing breakdown: " +
              f"prep={timing_breakdown['preparation']:.2f}s, " +
              f"api_call={timing_breakdown['api_call']:.2f}s, " +
              f"processing={timing_breakdown['processing']:.2f}s, " +
              f"total={timing_breakdown['total']:.2f}s")

        # Return the generated code and detailed timing information
        return generated_code, timing_breakdown['api_call'], timing_breakdown

    except Exception as e:
        # Record end time even for errors
        timing['api_call_end'] = time.time()
        timing['processing_end'] = time.time()

        # Log error with timing
        error_duration = timing['api_call_end'] - timing['api_call_start']
        print(f"API_ERROR_LOG: model={model}, time={time.time()}, error={str(e)}, duration={error_duration:.2f}s")

        # Re-raise the exception
        raise

def execute_code(code):
    """Execute the generated code and capture output"""
    import sys
    from io import StringIO
    import traceback
    import re

    # First, strip markdown code blocks if present
    print(f"DEBUG[sandbox]: Original code length: {len(code)} chars")

    # Extract Python code from within code blocks using regex
    python_code_blocks = re.findall(r'```python\s*(.*?)\s*```', code, re.DOTALL)

    if python_code_blocks:
        # Use the first Python code block if multiple are found
        code = python_code_blocks[0].strip()
    elif code.startswith('```python') or code.startswith('```'):
        # Fallback to original method if regex didn't work
        # Find the end of the opening code block
        start_idx = code.find('\n') + 1
        # Find the closing code block
        end_idx = code.rfind('```')
        if end_idx > start_idx:
            code = code[start_idx:end_idx].strip()
        else:
            code = code.replace('```python', '').replace('```', '').strip()

    print(f"DEBUG[sandbox]: Code after markdown stripping: {len(code)} chars")
    print(f"DEBUG[sandbox]: Code to execute: {code}")

    # Create a temporary file to execute directly
    import os
    import tempfile

    temp_fd, temp_path = tempfile.mkstemp(suffix='.py')
    with os.fdopen(temp_fd, 'w') as f:
        f.write(code)

    # Execute with subprocess to properly capture output
    import subprocess
    try:
        output = subprocess.check_output(["python3", temp_path],
                                      stderr=subprocess.STDOUT,
                                      text=True)
        execution_output = output.strip()
        execution_result = "Success"
        print(f"DEBUG[sandbox]: Direct execution output: '{execution_output}'")
    except subprocess.CalledProcessError as e:
        execution_output = e.output.strip()
        execution_result = f"Error: {e.returncode}"
        print(f"DEBUG[sandbox]: Direct execution error: '{execution_output}'")

    # Clean up temp file
    try:
        os.unlink(temp_path)
    except:
        pass

    return {
        "execution_output": execution_output,
        "execution_result": execution_result
    }

def check_environment():
    """Check and print environment variables for debugging"""
    print("DEBUG[sandbox]: Current environment variables:")
    print("DEBUG[sandbox]: GROQ_API_KEY:", "Present" if os.environ.get("GROQ_API_KEY") else "Missing")
    print("DEBUG[sandbox]: Environment keys:", list(os.environ.keys()))

def process_task(user_input):
    """Process a user task by generating and executing code"""
    check_environment()

    try:
        # Generate code
        try:
            import time
            code_gen_start = time.time()

            # Call generate_code with enhanced timing information
            # Returns: generated_code, api_call_time, detailed_timing
            generated_code, api_call_time, timing_breakdown = generate_code(user_input)

            # Get the total generation time (from our perspective)
            code_gen_total_time = time.time() - code_gen_start

            print(f"Successfully generated code for: {user_input}")
            print(f"DEBUG[sandbox]: API call time: {api_call_time:.2f}s, Total gen time: {code_gen_total_time:.2f}s")

            # Store the GROQ API time as the API call time specifically
            groq_time = api_call_time

            # Store detailed timing in a variable for later use
            timing_details = timing_breakdown

        except Exception as e:
            try:
                # Log API error with proper imports in this scope
                import time
                import os
                model_name = os.environ.get("GROQ_MODEL", "unknown")
                current_time = time.time()
                print(f"API_ERROR_LOG: model={model_name}, time={current_time}, error={str(e)}")
                print(f"Error generating code: {str(e)}")
            except Exception as inner_e:
                # Handle any errors in the error handler itself
                print(f"Error in error handler: {str(inner_e)}")
                print(f"Original error: {str(e)}")
            # If we can't generate code, use a default function that accomplishes nothing
            generated_code = f"""
# Fallback implementation since code generation failed
print("Code generation failed with error: {str(e).replace('"', '\\"')}")
print("Using fallback implementation")

def fallback_function():
    print("This is a fallback function since the original code could not be generated")
    return "Fallback completed"

result = fallback_function()
print(f"Result: {{result}}")
"""
            # Set timing values to 0 for error cases
            groq_time = 0
            code_gen_total_time = 0
            timing_details = {
                'preparation': 0,
                'api_call': 0,
                'processing': 0,
                'total': 0
            }

        # Save the generated code to a file in the home dir
        import os
        home_dir = os.path.expanduser("~")
        code_file_path = os.path.join(home_dir, "generated_code.py")
        with open(code_file_path, "w") as f:
            f.write(generated_code)

        # Execute the code
        execution_results = execute_code(generated_code)

        # Save the execution result to a file in the home dir
        output_file_path = os.path.join(home_dir, "execution_output.txt")
        with open(output_file_path, "w") as f:
            f.write(execution_results["execution_output"])

        # Return the results with detailed timing information
        import time
        result = {
            "generated_code": generated_code,
            "execution_output": execution_results["execution_output"],
            "execution_result": execution_results["execution_result"],
            
            # Timing information - backward compatible
            "groq_time": groq_time,                  # Time for the API call only
            "code_generation_time": code_gen_total_time,  # Total time for code generation (including API latency)
            
            # Enhanced detailed timing information
            "timing_details": timing_details,        # Detailed breakdown of API call timing
            
            # Add timestamp for tracking
            "sandbox_start_time": time.time()       # Timestamp when sandbox processing is complete
        }

        # Use explicit import in this scope to avoid UnboundLocalError
        import json as result_json
        result_file_path = os.path.join(home_dir, "result.json")
        with open(result_file_path, "w") as f:
            result_json.dump(result, f)

        return result
    except Exception as e:
        # Ensure all imports are within this scope
        import traceback
        import os
        import json
        import time
        
        # Get detailed error information
        error_trace = traceback.format_exc()
        print(f"DEBUG[sandbox]: Process task error: {str(e)}\n{error_trace}")
        
        # Create result with error details
        error_result = {
            "generated_code": f"Error occurred during processing: {str(e)}",
            "execution_output": error_trace,
            "execution_result": f"Error: {str(e)}",
            # Add timing fields to ensure the result has the expected structure
            "groq_time": 0,
            "code_generation_time": 0,
            "timing_details": {
                "preparation": 0,
                "api_call": 0,
                "processing": 0,
                "total": 0
            },
            "sandbox_start_time": time.time()
        }
        
        # Try to save the error result
        try:
            home_dir = os.path.expanduser("~")
            error_file_path = os.path.join(home_dir, "error_result.json")
            # Use explicit import in this scope to avoid UnboundLocalError
            import json as error_json
            with open(error_file_path, "w") as f:
                error_json.dump(error_result, f)
        except Exception as save_error:
            print(f"Error saving error result: {str(save_error)}")
            
        return error_result

# The following code runs when this script is executed directly
if __name__ == "__main__":
    # Import all needed modules at this scope
    import sys
    import json as main_json
    import os
    import time

    if len(sys.argv) > 1:
        user_input = sys.argv[1]
    else:
        user_input = input("Enter your task: ")

    result = process_task(user_input)
    print(main_json.dumps(result))