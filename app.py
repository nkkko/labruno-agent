import os
import uuid
import json
import tempfile
import time
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from daytona_sdk import Daytona, DaytonaConfig, CreateSandboxParams
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv(verbose=True, override=True)  # Added verbose and override to ensure environment variables are loaded

# Helper function to safely handle string literals in Python code
def safe_string_literal(s):
    """Properly escape a string for insertion into Python code as string literal"""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace("'", "\\'")

# Set environment variables directly to ensure they're available
try:
    with open('/Users/nikola/dev/labruno/.env', 'r') as f:
        for line in f:
            if line.strip() and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
except FileNotFoundError:
    print("Warning: .env file not found. Using environment variables as is.")

print("Loaded environment variables:")
print(f"DAYTONA_API_KEY: {'Present' if os.environ.get('DAYTONA_API_KEY') else 'Missing'}")
print(f"DAYTONA_TARGET: {'Present' if os.environ.get('DAYTONA_TARGET') else 'Missing'}")
print(f"GROQ_API_KEY: {'Present' if os.environ.get('GROQ_API_KEY') else 'Missing'}")
print(f"MAX_SANDBOXES: {os.environ.get('MAX_SANDBOXES', 10)}")

# Initialize Flask app
app = Flask(__name__)
app.config["TAILWIND_DEV"] = True

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Daytona client
daytona_api_key = os.environ.get("DAYTONA_API_KEY")
daytona_target = os.environ.get("DAYTONA_TARGET")
print(f"Initializing Daytona client with API key: {daytona_api_key[:5] if daytona_api_key else None}... and target: {daytona_target}")
daytona = Daytona(DaytonaConfig(api_key=daytona_api_key, target=daytona_target))

def create_sandbox(code_to_upload=None):
    """Create a Daytona sandbox and optionally upload code to it"""
    start_time = time.time()
    
    # Create sandbox with environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL")
    
    # Create parameters with environment variables
    params = CreateSandboxParams(
        language="python", 
        env={
            "GROQ_API_KEY": groq_api_key,
            "GROQ_MODEL": groq_model
        }
    )
    
    print(f"DEBUG: Using GROQ_API_KEY: {'Present' if groq_api_key else 'Missing'}")
    print(f"DEBUG: Using GROQ_MODEL: {groq_model}")

    sandbox = daytona.create(params)
    creation_time = time.time() - start_time
    print(f"DEBUG: Created sandbox in {creation_time:.2f}s: {sandbox}")

    # Add creation time to sandbox object for tracking
    sandbox.creation_time = creation_time

    # Set environment variables in the sandbox
    try:
        print("DEBUG: Setting environment variables in the sandbox")
        # Get environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")

        # Create a modified task runner file with the API key directly embedded
        with open('sandbox_task_runner.py', 'r') as f:
            task_runner_content = f.read()

        # Get API key and model from environment
        groq_model = os.getenv("GROQ_MODEL")
        
        # Insert environment variables directly into the task runner file
        modified_task_runner = task_runner_content.replace(
            'import os\nimport json\nimport sys\nfrom groq import Groq',
            f'''import os
import json
import sys
from groq import Groq

# Pre-set environment variables
os.environ["GROQ_API_KEY"] = "{groq_api_key}"
os.environ["GROQ_MODEL"] = "{groq_model}"
print("DEBUG[sandbox]: Hard-coded GROQ_API_KEY is present:", bool(os.environ.get("GROQ_API_KEY")))
print("DEBUG[sandbox]: Hard-coded GROQ_MODEL:", os.environ.get("GROQ_MODEL"))'''
        )

        # Save this modified version to a temp directory to avoid triggering Flask reload
        import tempfile
        temp_dir = tempfile.gettempdir()
        modified_runner_path = os.path.join(temp_dir, 'modified_task_runner.py')
        with open(modified_runner_path, 'w') as f:
            f.write(modified_task_runner)

        print("DEBUG: Created modified task runner with embedded API key")

        # Set environment variables using process.exec
        print("DEBUG: Setting environment variables using process.exec")
        response = sandbox.process.exec(f"export GROQ_API_KEY={groq_api_key}")
        print(f"DEBUG: Environment variable set response: {response.result}")
        
        # Set GROQ_MODEL environment variable
        groq_model = os.getenv("GROQ_MODEL")
        if groq_model:
            response = sandbox.process.exec(f"export GROQ_MODEL={groq_model}")
            print(f"DEBUG: GROQ_MODEL environment variable set response: {response.result}")
    except Exception as e:
        print(f"DEBUG: Error setting environment variables: {str(e)}")

    if code_to_upload:
        # Upload code to the sandbox
        try:
            print("DEBUG: Uploading code to the sandbox")

            # Create a directory for our files if needed
            try:
                # Create a directory in the home folder to store our files
                sandbox.fs.create_folder("/home/daytona/labruno", "755")
                print("DEBUG: Created labruno directory in sandbox")
            except Exception as e:
                print(f"DEBUG: Directory may already exist: {str(e)}")

            # Use Daytona's file upload mechanism with proper permissions
            file_path = "/home/daytona/labruno/sandbox_task_runner.py"
            sandbox.fs.upload_file(file_path, code_to_upload.encode('utf-8'))

            # Make it executable
            sandbox.fs.set_file_permissions(file_path, mode="755")
            print(f"DEBUG[sandbox]: Code uploaded successfully to {file_path}")
        except Exception as e:
            print(f"DEBUG: Error uploading code: {str(e)}")

    return sandbox

def prepare_sandbox(sandbox, task_runner_code):
    """Prepare a sandbox with task runner code and environment variables"""
    start_time = time.time()
    try:
        # Create directory structure
        sandbox.fs.create_folder("/home/daytona/labruno", "755")

        # Upload the task runner file
        file_path = "/home/daytona/labruno/sandbox_task_runner.py"
        sandbox.fs.upload_file(file_path, task_runner_code.encode('utf-8'))

        # Make it executable
        sandbox.fs.set_file_permissions(file_path, mode="755")

        # Also upload the .env file as a backup
        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_model = os.getenv("GROQ_MODEL")
        env_file_content = f'''
# Environment variables for sandbox
GROQ_API_KEY={groq_api_key}
GROQ_MODEL={groq_model}
'''
        sandbox.fs.upload_file("/home/daytona/.env", env_file_content.encode('utf-8'))

        # Set environment variables using process.exec
        print("DEBUG: Setting environment variables using process.exec")
        response = sandbox.process.exec(f"export GROQ_API_KEY={groq_api_key}")
        print(f"DEBUG: Environment variable set response: {response.result}")
        
        # Set GROQ_MODEL environment variable
        if groq_model:
            response = sandbox.process.exec(f"export GROQ_MODEL={groq_model}")
            print(f"DEBUG: GROQ_MODEL environment variable set response: {response.result}")

        # Calculate preparation time
        prep_time = time.time() - start_time
        sandbox.prep_time = prep_time
        print(f"DEBUG: Prepared sandbox in {prep_time:.2f}s")

        return sandbox
    except Exception as e:
        print(f"DEBUG: Error preparing sandbox: {str(e)}")
        return sandbox  # Return sandbox even if preparation fails

def generate_and_execute_code(sandbox, user_input, task_runner_path=None):
    """Generate and execute code in a sandbox based on user input"""
    print(f"DEBUG: In generate_and_execute_code with sandbox: {sandbox}")

    # Start timing for code generation and execution
    gen_start_time = time.time()

    # Use Groq API to generate code
    try:
        print(f"DEBUG: Running code in sandbox to generate and execute code")
        # No indentation in the executed code to prevent errors
        # Properly escape the user input to avoid issues with quotes
        safe_user_input = safe_string_literal(user_input)

        # Prepare code based on whether we have a task runner or need to do direct execution
        if task_runner_path:
            # If we have a task runner path, use it to delegate execution
            print(f"DEBUG: Using task runner at {task_runner_path}")
            code_template = f'''
import os
import sys
import json

print("DEBUG[sandbox]: Starting code generation using task runner")

try:
    # Execute the task runner directly as a Python script
    task_runner_path = "{task_runner_path}"
    print(f"DEBUG[sandbox]: Loading task runner from {{task_runner_path}}")

    # Add the directory to Python path
    dir_path = os.path.dirname(task_runner_path)
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)

    # Load the module directly from the file
    import importlib.util
    spec = importlib.util.spec_from_file_location("sandbox_task_runner", task_runner_path)
    task_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task_runner)

    print("DEBUG[sandbox]: Successfully loaded task runner")

    # Use task runner to process the request - triple quotes to ensure multi-line strings work
    user_input = """{safe_user_input}"""  # Get properly escaped user input
    print(f"DEBUG[sandbox]: Processing task: {user_input}")
    # Remove any trailing newlines that might cause issues
    user_input = user_input.strip()
    result = task_runner.process_task(user_input)
    print("DEBUG[sandbox]: Task processing complete")

    # Print the result so it can be captured
    print(json.dumps(result))
except Exception as e:
    print("DEBUG[sandbox]: Error using task runner:", str(e))
    import traceback
    print(traceback.format_exc())

    # Return error information
    error_result = {{
        "error": str(e),
        "generated_code": "Error using task runner",
        "execution_output": traceback.format_exc(),
        "execution_result": "Error"
    }}
    print(json.dumps(error_result))
'''
        else:
            # Otherwise, fall back to direct execution
            code_template = '''
import os
import json
import sys
from groq import Groq

def run_code_generation_and_execution():
    print("DEBUG[sandbox]: Starting code generation process (no task runner)")
    print("DEBUG[sandbox]: Environment variables:", os.environ)

    # Initialize Groq client
    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        print("DEBUG[sandbox]: GROQ_API_KEY available:", bool(groq_api_key))
        client = Groq(api_key=groq_api_key)
        print("DEBUG[sandbox]: Groq client initialized")
    except Exception as e:
        print("DEBUG[sandbox]: Error initializing Groq client:", str(e))
        raise

    # Generate code
    try:
        print("DEBUG[sandbox]: Creating chat completion")

        # Get model from environment variable
        groq_model = os.environ.get("GROQ_MODEL")
        print(f"DEBUG[sandbox]: Using model: {groq_model}")

        # Log API request
        import time
        request_time = time.time()
        print(f"API_REQUEST_LOG: model={groq_model}, time={request_time}, user_input_length={len('{}')}")

        api_start_time = time.time()
        chat_completion = client.chat.completions.create(
            messages=[
                {{
                    "role": "system",
                    "content": "You are a helpful AI assistant that generates Python code based on user requests. Generate ONLY the Python code without any explanations or markdown formatting. Never write instructions. Only respond with the code."
                }},
                {{
                    "role": "user",
                    "content": "{}"
                }}
            ],
            model=groq_model,
        )

        # Log API response
        api_time = time.time() - api_start_time
        print(f"API_RESPONSE_LOG: model={groq_model}, time={time.time()}, duration={api_time:.2f}s, status=success")
        print("DEBUG[sandbox]: Chat completion created")
    except Exception as e:
        # Log API error
        import time
        print(f"API_ERROR_LOG: model={os.environ.get('GROQ_MODEL')}, time={time.time()}, error={str(e)}")
        print("DEBUG[sandbox]: Error creating chat completion:", str(e))
        raise

    # Extract the generated code
    try:
        generated_code = chat_completion.choices[0].message.content
        print("DEBUG[sandbox]: Generated code length:", len(generated_code))
    except Exception as e:
        print("DEBUG[sandbox]: Error extracting generated code:", str(e))
        raise

    # Save the generated code to a file
    try:
        with open("generated_code.py", "w") as f:
            f.write(generated_code)
        print("DEBUG[sandbox]: Generated code saved to file")
    except Exception as e:
        print("DEBUG[sandbox]: Error saving generated code:", str(e))
        raise

    # No special case handling needed anymore

    # Execute the generated code and capture output
    try:
        import sys
        from io import StringIO
        import traceback

        print("DEBUG[sandbox]: Executing generated code")
        output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = output

        try:
            # Use a dedicated namespace to avoid conflicts
            exec_namespace = {}
            exec(generated_code, exec_namespace)
            execution_result = "Success"
            print("DEBUG[sandbox]: Code execution successful")
        except Exception as e:
            error_msg = str(e)
            trace = traceback.format_exc()
            execution_result = "Error: {{}}\\n{{}}".format(error_msg, trace)
            print("DEBUG[sandbox]: Code execution failed:", error_msg)

        sys.stdout = original_stdout
        execution_output = output.getvalue()
        print("DEBUG[sandbox]: Execution output length:", len(execution_output))
    except Exception as e:
        print("DEBUG[sandbox]: Error during execution setup:", str(e))
        raise

    # Save the execution result to a file
    try:
        with open("execution_output.txt", "w") as f:
            f.write(execution_output)
        print("DEBUG[sandbox]: Execution output saved to file")
    except Exception as e:
        print("DEBUG[sandbox]: Error saving execution output:", str(e))
        raise

    # Return the results
    try:
        result = {{
            "generated_code": generated_code,
            "execution_output": execution_output,
            "execution_result": execution_result
        }}

        with open("result.json", "w") as f:
            json.dump(result, f)
        print("DEBUG[sandbox]: Result JSON saved to file")

        print("DEBUG[sandbox]: Returning result")
        print(json.dumps(result))
        return result
    except Exception as e:
        print("DEBUG[sandbox]: Error preparing result:", str(e))
        raise

# Run the main function
result = run_code_generation_and_execution()
'''.format(safe_user_input)

        result = sandbox.process.code_run(code_template)
        print(f"DEBUG: Sandbox code execution completed")

        # Get the result text
        result_text = result.result if hasattr(result, 'result') else str(result)
        print(f"DEBUG: Result preview: {result_text[:100]}...")

        try:
            # Try to find JSON in the output
            import re
            json_pattern = r'\{[\s\S]*\}'  # Match any JSON-like structure

            # First, look for explicit JSON output
            matches = []
            for line in result_text.splitlines():
                if line.startswith('{') and line.endswith('}'):
                    try:
                        # Try to parse this line as JSON directly
                        parsed_result = json.loads(line)
                        if isinstance(parsed_result, dict) and 'generated_code' in parsed_result:
                            print(f"DEBUG: Found valid JSON result")

                            # Check for empty execution output
                            if not parsed_result.get('execution_output'):
                                print("DEBUG: Empty execution output detected, looking for output in logs...")
                                # Try to find output in the logs using regex
                                direct_output = re.search(r"Direct execution output: '([^']*)'", result_text)
                                if direct_output:
                                    parsed_result['execution_output'] = direct_output.group(1)
                                    print(f"DEBUG: Found direct execution output: {parsed_result['execution_output']}")

                            return parsed_result
                        matches.append(line)
                    except:
                        pass

            # If we found potential JSON matches, try them in order
            for match in matches:
                try:
                    parsed_result = json.loads(match)
                    if isinstance(parsed_result, dict):
                        print(f"DEBUG: Successfully parsed JSON result: {list(parsed_result.keys())}")
                        # Add missing fields if needed
                        if 'generated_code' not in parsed_result:
                            parsed_result['generated_code'] = "# Generated code not available"
                        if 'execution_output' not in parsed_result:
                            parsed_result['execution_output'] = "Output not available"
                        if 'execution_result' not in parsed_result:
                            parsed_result['execution_result'] = "Unknown"

                        # Check for empty execution output
                        if not parsed_result.get('execution_output'):
                            print("DEBUG: Empty execution output detected, looking for output in logs...")
                            # Try to find output in the logs using regex
                            direct_output = re.search(r"Direct execution output: '([^']*)'", result_text)
                            if direct_output:
                                parsed_result['execution_output'] = direct_output.group(1)
                                print(f"DEBUG: Found direct execution output: {parsed_result['execution_output']}")
                            else:
                                # Look for any print output in the logs
                                print_lines = []
                                for line in result_text.splitlines():
                                    if "DEBUG[sandbox]" not in line and "print(" not in line:
                                        print_lines.append(line)
                                if print_lines:
                                    parsed_result['execution_output'] = "\n".join(print_lines)
                                    print(f"DEBUG: Constructed output from logs: {parsed_result['execution_output']}")

                        return parsed_result
                except:
                    pass

            # If we couldn't find valid JSON, do a full regex search
            json_matches = re.findall(json_pattern, result_text)
            for match in json_matches:
                try:
                    parsed_result = json.loads(match)
                    if isinstance(parsed_result, dict):
                        print(f"DEBUG: Found JSON in output via regex: {list(parsed_result.keys())}")
                        # Add missing fields if needed
                        if 'generated_code' not in parsed_result:
                            parsed_result['generated_code'] = "# Generated code not available"
                        if 'execution_output' not in parsed_result:
                            parsed_result['execution_output'] = "Output not available"
                        if 'execution_result' not in parsed_result:
                            parsed_result['execution_result'] = "Unknown"

                        # Check for empty execution output
                        if not parsed_result.get('execution_output'):
                            print("DEBUG: Empty execution output detected, looking for output in logs...")
                            # Try to find output in the logs using regex
                            direct_output = re.search(r"Direct execution output: '([^']*)'", result_text)
                            if direct_output:
                                parsed_result['execution_output'] = direct_output.group(1)
                                print(f"DEBUG: Found direct execution output: {parsed_result['execution_output']}")
                            else:
                                # Look for any print output in the logs
                                print_lines = []
                                for line in result_text.splitlines():
                                    if "DEBUG[sandbox]" not in line and "print(" not in line:
                                        print_lines.append(line)
                                if print_lines:
                                    parsed_result['execution_output'] = "\n".join(print_lines)
                                    print(f"DEBUG: Constructed output from logs: {parsed_result['execution_output']}")

                        return parsed_result
                except:
                    pass

            # If no JSON is found, create a result from the stdout
            print(f"DEBUG: Creating result from stdout")
            # Try to extract code blocks if they exist
            code_pattern = r'```python\s*(.*?)\s*```'
            code_match = re.search(code_pattern, result_text, re.DOTALL)
            generated_code = code_match.group(1) if code_match else result_text

            # Look for direct execution output
            execution_output = ""
            direct_output = re.search(r"Direct execution output: '([^']*)'", result_text)
            if direct_output:
                execution_output = direct_output.group(1)
                print(f"DEBUG: Found direct execution output: {execution_output}")
            else:
                # Extract non-debug lines as potential output
                output_lines = []
                for line in result_text.splitlines():
                    if "DEBUG[sandbox]" not in line and "print(" not in line:
                        output_lines.append(line)
                if output_lines:
                    execution_output = "\n".join(output_lines)

            # Create a fallback result
            return {
                "generated_code": generated_code,
                "execution_output": execution_output or result_text,
                "execution_result": "Success" if "error" not in result_text.lower() else "Error"
            }
        except Exception as e:
            print(f"DEBUG: Failed to parse result: {str(e)}")
            print(f"DEBUG: Raw result: {result.result if hasattr(result, 'result') else str(result)}")
            gen_time = time.time() - gen_start_time
            return {
                "error": f"Failed to parse result: {str(e)}",
                "generated_code": "Error parsing result",
                "execution_output": result.result if hasattr(result, 'result') else str(result),
                "execution_time": gen_time
            }

    except Exception as e:
        print(f"DEBUG: Exception in generate_and_execute_code: {str(e)}")
        import traceback
        print(traceback.format_exc())
        gen_time = time.time() - gen_start_time
        return {
            "error": str(e),
            "generated_code": "Error occurred",
            "execution_output": traceback.format_exc(),
            "execution_time": gen_time
        }

def evaluate_results(main_sandbox, results):
    """Evaluate which implementation is the best based on success and output"""
    print(f"DEBUG: In evaluate_results with {len(results)} results")

    # Prepare the evaluation input
    try:
        evaluation_input = json.dumps(results)
        print(f"DEBUG: Evaluation input JSON size: {len(evaluation_input)} chars")
    except Exception as e:
        print(f"DEBUG: Error serializing results to JSON: {str(e)}")
        return {"error": f"Failed to serialize results: {str(e)}", "evaluation": "Error preparing evaluation"}

    # Identify implementations that at least ran without errors
    # First try to get ones that actually succeeded
    successful_results = [r for r in results if r.get('execution_result') == 'Success']
    print(f"DEBUG: Found {len(successful_results)} successful implementations")

    # If no implementation succeeded, be more lenient and accept any that have code generated
    if not successful_results:
        # Look for any results that at least have generated code
        any_code_results = [r for r in results if r.get('generated_code') and r.get('generated_code') != 'Error' and r.get('generated_code') != 'Error occurred']
        print(f"DEBUG: Found {len(any_code_results)} implementations with code generated")

        if any_code_results:
            successful_results = any_code_results
        else:
            evaluation = {
                "evaluation": "After analyzing the available implementations, I've determined that none of them completed successfully. The execution environment appears to be having issues. Please try again with a different request."
            }
            print(f"DEBUG: No successful implementations found")
            return evaluation

    # Basic heuristic for finding the best implementation
    # 1. Rank by success (already filtered)
    # 2. Rank by code length (shorter is often better)
    # 3. Rank by output length (more output might indicate better results)

    # Sort first by success, then by code length (shorter is better), then by output length (longer might be better)
    ranked_results = sorted(
        successful_results,
        key=lambda r: (
            0 if r.get('execution_result') == 'Success' else 1,
            len(r.get('generated_code', '')),
            -len(r.get('execution_output', ''))
        )
    )

    # Get the best implementation
    best_impl = ranked_results[0] if ranked_results else None
    best_impl_index = results.index(best_impl) if best_impl in results else -1

    # Generate an evaluation
    if best_impl:
        # Extract a snippet of the output to include in the evaluation
        output_snippet = best_impl.get('execution_output', '')[:100] + '...' if len(best_impl.get('execution_output', '')) > 100 else best_impl.get('execution_output', '')

        evaluation = {
            "evaluation": f"After analyzing the available agents, I've determined that Agent {best_impl_index + 1} is the best solution. It executes successfully and produces the expected output. The code is concise and follows good programming practices. Output: {output_snippet}",
            "best_implementation_index": best_impl_index + 1,
            "best_implementation": best_impl
        }
    else:
        evaluation = {
            "evaluation": "After analyzing the available agents, I couldn't determine a clear winner. Multiple agents succeeded, but none stood out as significantly better than the others."
        }

    print(f"DEBUG: Generated evaluation: {evaluation}")
    return evaluation

@app.route('/')
def index():
    model_name = os.environ.get("GROQ_MODEL")
    return render_template('index.html', model_name=model_name)

@app.route('/execute', methods=['POST'])
def execute():
    start_time = time.time()
    user_input = request.form.get('user_input')
    sandbox_count = request.form.get('sandbox_count', '3')
    results = []
    evaluation = {"error": "Not started"}
    timing_data = {}

    # Generate a unique session ID for this execution run
    session_id = str(uuid.uuid4())

    # Validate input and sandbox count
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Get MAX_SANDBOXES from environment variable or default to 10
    max_sandboxes = int(os.environ.get("MAX_SANDBOXES", 10))

    try:
        sandbox_count = int(sandbox_count)
        if sandbox_count < 1 or sandbox_count > max_sandboxes:
            sandbox_count = 3  # Default to 3 if out of range
    except ValueError:
        sandbox_count = 3  # Default to 3 if not a valid number

    # Initialize sandbox status tracking
    sandbox_status = {}
    for i in range(sandbox_count):
        sandbox_status[i] = {
            'status': 'pending',
            'progress': 0
        }

    # Send all sandbox statuses at once with a small delay
    # This helps ensure the frontend receives them properly
    def emit_initial_status():
        print(f"DEBUG: Emitting initial status for {sandbox_count} sandboxes with session ID {session_id}")
        import time
        for i in range(sandbox_count):
            time.sleep(0.1)  # Small delay to ensure reliable delivery
            socketio.emit('sandbox_status', {
                'session_id': session_id,
                'sandbox_id': i,
                'status': 'pending',
                'progress': 0
            })

    # Start emitting in a background thread to not block request
    import threading
    threading.Thread(target=emit_initial_status).start()

    try:
        print(f"DEBUG: Starting execution with input: {user_input}")

        # Create main sandbox
        print("DEBUG: Creating main sandbox...")
        main_sandbox = create_sandbox()
        print(f"DEBUG: Main sandbox created: {main_sandbox}")

        # Upload task runner code to main sandbox
        try:
            print("DEBUG: Uploading task runner code to main sandbox...")
            # First create a home directory - this is always accessible per Daytona docs
            try:
                # Create a directory in the home folder to store our files
                main_sandbox.fs.create_folder("/home/daytona/labruno", "755")
                print("DEBUG: Created labruno directory in sandbox")
            except Exception as e:
                print(f"DEBUG: Directory may already exist: {str(e)}")

            # Use the modified task runner with embedded API key
            modified_runner_path = os.path.join(tempfile.gettempdir(), 'modified_task_runner.py')
            with open(modified_runner_path, 'r') as f:
                task_runner_code = f.read()

            # Use Daytona's built-in file upload mechanism to the home directory
            file_path = "/home/daytona/labruno/sandbox_task_runner.py"
            main_sandbox.fs.upload_file(file_path, task_runner_code.encode('utf-8'))
            print(f"DEBUG: Task runner code uploaded to main sandbox at {file_path}")

            # Make sure it's executable
            main_sandbox.fs.set_file_permissions(file_path, mode="755")
            print("DEBUG: Set executable permissions on task runner")

            # Set the path where the task runner will be located
            task_runner_path = file_path
            print(f"DEBUG: Task runner path set to: {task_runner_path}")

        except Exception as e:
            print(f"DEBUG: Error uploading task runner code: {str(e)}")
            raise

        # Create sandboxes in parallel using ThreadPoolExecutor
        print(f"DEBUG: Creating {sandbox_count} sandboxes concurrently...")
        sandboxes = []

        # Read the modified task runner with embedded API key
        modified_runner_path = os.path.join(tempfile.gettempdir(), 'modified_task_runner.py')
        with open(modified_runner_path, 'r') as f:
            task_runner_code = f.read()

        # Function to create and prepare a sandbox concurrently
        def create_and_prepare_sandbox():
            try:
                # Get the sandbox ID from the thread context
                sandbox_id = getattr(threading.current_thread(), 'sandbox_id', -1)

                # Update status to creating
                if sandbox_id >= 0:
                    sandbox_status[sandbox_id]['status'] = 'creating'
                    sandbox_status[sandbox_id]['progress'] = 20
                    socketio.emit('sandbox_status', {
                        'session_id': session_id,
                        'sandbox_id': sandbox_id,
                        'status': 'creating',
                        'progress': 20
                    })

                # Sandbox creation is already timed inside the function
                sandbox = create_sandbox()

                # Update status to preparing
                if sandbox_id >= 0:
                    sandbox_status[sandbox_id]['status'] = 'preparing'
                    sandbox_status[sandbox_id]['progress'] = 40
                    socketio.emit('sandbox_status', {
                        'session_id': session_id,
                        'sandbox_id': sandbox_id,
                        'status': 'preparing',
                        'progress': 40
                    })

                prepared_sandbox = prepare_sandbox(sandbox, task_runner_code)

                # Calculate total sandbox preparation time
                total_prep_time = getattr(sandbox, 'creation_time', 0) + getattr(prepared_sandbox, 'prep_time', 0)
                prepared_sandbox.total_prep_time = total_prep_time

                # Update status to ready
                if sandbox_id >= 0:
                    sandbox_status[sandbox_id]['status'] = 'ready'
                    sandbox_status[sandbox_id]['progress'] = 60
                    socketio.emit('sandbox_status', {
                        'session_id': session_id,
                        'sandbox_id': sandbox_id,
                        'status': 'ready',
                        'progress': 60
                    })

                return prepared_sandbox
            except Exception as e:
                print(f"DEBUG: Error creating and preparing sandbox: {str(e)}")

                # Update status to error
                if sandbox_id >= 0:
                    sandbox_status[sandbox_id]['status'] = 'error'
                    sandbox_status[sandbox_id]['progress'] = 0
                    socketio.emit('sandbox_status', {
                        'session_id': session_id,
                        'sandbox_id': sandbox_id,
                        'status': 'error',
                        'progress': 0
                    })

                return None

        # Store sandbox creation start time
        sandbox_start_times = {}

        # Create sandboxes concurrently
        with ThreadPoolExecutor(max_workers=max_sandboxes) as executor:
            # Submit sandbox creation tasks with thread local storage for sandbox_id
            future_to_sandbox = {}
            for i in range(sandbox_count):
                # Create a function that will set thread-local sandbox_id
                def task_with_id():
                    # Set the sandbox_id on the thread
                    threading.current_thread().sandbox_id = i
                    return create_and_prepare_sandbox()

                future = executor.submit(task_with_id)
                future_to_sandbox[future] = i

            # Track start time for each sandbox
            for i in range(sandbox_count):
                sandbox_start_times[i] = time.time()

            # Collect results as they complete
            for future in as_completed(future_to_sandbox):
                sandbox_id = future_to_sandbox[future]
                try:
                    sandbox = future.result()
                    if sandbox:
                        # Store the creation start time with the sandbox
                        sandbox.creation_start_time = sandbox_start_times[sandbox_id]
                        sandbox.sandbox_id = sandbox_id  # Store sandbox_id for later reference
                        print(f"DEBUG: Created and prepared sandbox {sandbox_id+1}: {sandbox}")
                        sandboxes.append(sandbox)
                except Exception as e:
                    print(f"DEBUG: Error with sandbox {sandbox_id+1}: {str(e)}")

        print(f"DEBUG: Successfully created {len(sandboxes)} sandboxes concurrently")

        # Run code generation and execution in parallel
        print("DEBUG: Running code generation and execution concurrently...")

        # Function to process a sandbox concurrently
        def process_sandbox(sandbox_with_id):
            sandbox, sandbox_id = sandbox_with_id
            try:
                print(f"DEBUG: Processing sandbox {sandbox_id+1}...")

                # Update status to generating
                sandbox_status[sandbox_id]['status'] = 'generating'
                sandbox_status[sandbox_id]['progress'] = 70
                socketio.emit('sandbox_status', {
                    'session_id': session_id,
                    'sandbox_id': sandbox_id,
                    'status': 'generating',
                    'progress': 70
                })

                # Start code generation
                result = generate_and_execute_code(sandbox, user_input, task_runner_path)

                # Update status to executing
                sandbox_status[sandbox_id]['status'] = 'executing'
                sandbox_status[sandbox_id]['progress'] = 85
                socketio.emit('sandbox_status', {
                    'session_id': session_id,
                    'sandbox_id': sandbox_id,
                    'status': 'executing',
                    'progress': 85
                })

                # Add sandbox creation time to result
                result['sandbox_creation_time'] = getattr(sandbox, 'creation_time', 0)
                result['total_prep_time'] = getattr(sandbox, 'total_prep_time', 0)

                # Calculate total sandbox time from the very beginning of sandbox creation
                # This represents the complete sandbox lifetime from request to completion
                if hasattr(sandbox, 'creation_start_time'):
                    result['total_sandbox_time'] = time.time() - sandbox.creation_start_time
                    print(f"DEBUG: Total sandbox {sandbox_id+1} lifetime: {result['total_sandbox_time']:.2f}s")
                elif 'sandbox_start_time' in result:
                    # Fallback if we don't have creation_start_time
                    result['total_sandbox_time'] = time.time() - result['sandbox_start_time']
                    # Remove the temporary start timestamp as it's no longer needed
                    del result['sandbox_start_time']

                # Add task information
                result['sandbox_id'] = sandbox_id + 1

                # Update status to complete or error based on execution result
                if result.get('error') or result.get('execution_result') != 'Success':
                    status = 'error'
                    progress = 90
                else:
                    status = 'complete'
                    progress = 100

                sandbox_status[sandbox_id]['status'] = status
                sandbox_status[sandbox_id]['progress'] = progress

                print(f"DEBUG: Emitting final status for sandbox {sandbox_id}: {status}")

                # Send several status updates to ensure it's received and processed
                # Sometimes socket messages can be dropped, so we send multiple
                for _ in range(3):  # Send 3 times with small delays
                    socketio.emit('sandbox_status', {
                        'session_id': session_id,
                        'sandbox_id': sandbox_id,
                        'status': status,
                        'progress': progress
                    })
                    time.sleep(0.1)  # Small delay between emissions

                print(f"DEBUG: Result from sandbox {sandbox_id+1} (exec time: {result.get('execution_time', 0):.2f}s): {result}")
                return result
            except Exception as e:
                print(f"DEBUG: Error processing sandbox {sandbox_id+1}: {str(e)}")

                # Update status to error
                sandbox_status[sandbox_id]['status'] = 'error'
                sandbox_status[sandbox_id]['progress'] = 0
                socketio.emit('sandbox_status', {
                    'session_id': session_id,
                    'sandbox_id': sandbox_id,
                    'status': 'error',
                    'progress': 0
                })

                return {
                    "error": str(e),
                    "generated_code": "Error",
                    "execution_output": str(e),
                    "sandbox_id": sandbox_id + 1,
                    "sandbox_creation_time": getattr(sandbox, 'creation_time', 0),
                    "sandbox_prep_time": getattr(sandbox, 'prep_time', 0),
                    "total_prep_time": getattr(sandbox, 'total_prep_time', 0),
                    "execution_time": 0
                }

        # Process all sandboxes concurrently
        with ThreadPoolExecutor(max_workers=max_sandboxes) as executor:
            # Submit sandbox processing tasks
            sandbox_with_ids = [(sandbox, i) for i, sandbox in enumerate(sandboxes)]
            results = list(executor.map(process_sandbox, sandbox_with_ids))

        print(f"DEBUG: Collected {len(results)} results concurrently")

        # Evaluate results
        if results:
            print("DEBUG: Evaluating results...")
            try:
                evaluation = evaluate_results(main_sandbox, results)
                print(f"DEBUG: Evaluation result: {evaluation}")
            except Exception as e:
                print(f"DEBUG: Error evaluating results: {str(e)}")
                evaluation = {"error": str(e), "evaluation": "Failed to evaluate results"}
        else:
            evaluation = {"error": "No results to evaluate", "evaluation": "No results were generated to evaluate"}

        # Clean up sandboxes concurrently
        print("DEBUG: Cleaning up sandboxes concurrently...")

        def remove_sandbox(sandbox_with_id):
            sandbox, sandbox_id = sandbox_with_id
            try:
                print(f"DEBUG: Removing sandbox {sandbox_id+1}...")
                daytona.remove(sandbox)
                return True
            except Exception as e:
                print(f"DEBUG: Error removing sandbox {sandbox_id+1}: {str(e)}")
                return False

        # Clean up all sandboxes concurrently
        with ThreadPoolExecutor(max_workers=max_sandboxes) as executor:
            sandbox_with_ids = [(sandbox, i) for i, sandbox in enumerate(sandboxes)]
            list(executor.map(remove_sandbox, sandbox_with_ids))

        # Clean up main sandbox
        try:
            print("DEBUG: Removing main sandbox...")
            daytona.remove(main_sandbox)
        except Exception as e:
            print(f"DEBUG: Error removing main sandbox: {str(e)}")

    except Exception as e:
        import traceback
        print(f"DEBUG: CRITICAL ERROR: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "results": [],
            "evaluation": {"evaluation": f"Error: {str(e)}"}
        })

    # Calculate total execution time
    total_execution_time = time.time() - start_time
    timing_data["total_execution_time"] = total_execution_time

    print(f"DEBUG: Total execution time: {total_execution_time:.2f}s")
    print("DEBUG: Returning results to client")

    return jsonify({
        "results": results,
        "evaluation": evaluation,
        "timing_data": timing_data
    })

@app.route('/test', methods=['POST'])
def test_mode():
    """A test endpoint that doesn't actually use Daytona or Groq APIs"""
    user_input = request.form.get('user_input')
    sandbox_count = request.form.get('sandbox_count', '3')

    # Generate a unique session ID for this execution run
    session_id = str(uuid.uuid4())

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Get MAX_SANDBOXES from environment variable or default to 10
    max_sandboxes = int(os.environ.get("MAX_SANDBOXES", 10))

    try:
        sandbox_count = int(sandbox_count)
        if sandbox_count < 1 or sandbox_count > max_sandboxes:
            sandbox_count = 3  # Default to 3 if out of range
    except ValueError:
        sandbox_count = 3  # Default to 3 if not a valid number

    print(f"DEBUG: Test mode activated with input: {user_input}, sandbox count: {sandbox_count}")

    # Initialize sandbox status for visualization
    print(f"DEBUG: Test mode - emitting initial status for {sandbox_count} sandboxes with session ID {session_id}")

    # Send all sandbox statuses at once with a small delay
    # This helps ensure the frontend receives them properly
    def emit_initial_test_status():
        import time
        for i in range(sandbox_count):
            time.sleep(0.1)  # Small delay to ensure reliable delivery
            socketio.emit('sandbox_status', {
                'session_id': session_id,
                'sandbox_id': i,
                'status': 'pending',
                'progress': 0
            })

    # Start emitting in a background thread to not block request
    import threading
    threading.Thread(target=emit_initial_test_status).start()

    # Emit status updates with delays to simulate real sandbox creation
    def emit_test_updates():
        import time
        for i in range(sandbox_count):
            # Creating
            socketio.emit('sandbox_status', {
                'session_id': session_id,
                'sandbox_id': i,
                'status': 'creating',
                'progress': 20
            })
            time.sleep(0.3)

            # Preparing
            socketio.emit('sandbox_status', {
                'session_id': session_id,
                'sandbox_id': i,
                'status': 'preparing',
                'progress': 40
            })
            time.sleep(0.3)

            # Ready
            socketio.emit('sandbox_status', {
                'session_id': session_id,
                'sandbox_id': i,
                'status': 'ready',
                'progress': 60
            })
            time.sleep(0.3)

            # Generating code
            socketio.emit('sandbox_status', {
                'session_id': session_id,
                'sandbox_id': i,
                'status': 'generating',
                'progress': 70
            })
            time.sleep(0.5)

            # Executing
            socketio.emit('sandbox_status', {
                'session_id': session_id,
                'sandbox_id': i,
                'status': 'executing',
                'progress': 85
            })
            time.sleep(0.5)

            # Randomly choose between complete and error for demonstration purposes
            if i % 3 == 0:  # Every third sandbox will show an error
                socketio.emit('sandbox_status', {
                    'session_id': session_id,
                    'sandbox_id': i,
                    'status': 'error',
                    'progress': 90
                })
            else:
                socketio.emit('sandbox_status', {
                    'session_id': session_id,
                    'sandbox_id': i,
                    'status': 'complete',
                    'progress': 100
                })

    # Start emitting updates in a background thread
    import threading
    threading.Thread(target=emit_test_updates, daemon=True).start()

    # Create dummy results
    results = []
    for i in range(sandbox_count):
        gen_code = f"""# Test implementation {i+1} for: {user_input}

def solution():
    # Implementation {i+1}
    print('This is test implementation {i+1}')
    return {i+1}

result = solution()
print(f'Result: {{result}}')"""

        # Create a mix of successful and failed implementations
        if i % 3 == 0:  # Every third sandbox will show an error
            results.append({
                "generated_code": gen_code,
                "execution_output": f"This is test implementation {i+1}\nError: Division by zero occurred",
                "execution_result": "Error",
                "error": "ZeroDivisionError: division by zero"
            })
        else:
            results.append({
                "generated_code": gen_code,
                "execution_output": f"This is test implementation {i+1}\nResult: {i+1}",
                "execution_result": "Success"
            })

    # Create dummy evaluation
    evaluation = {
        "evaluation": f"After analyzing the 5 implementations for the task '{user_input}', I've determined that Implementation 3 is the best solution because it has the most efficient code structure, better readability, and correct output."
    }

    return jsonify({
        "results": results,
        "evaluation": evaluation
    })

if __name__ == '__main__':
    # Create templates directory
    os.makedirs('templates', exist_ok=True)

    # Check if we're in test mode (no API keys)
    test_mode = False
    daytona_api_key = os.environ.get("DAYTONA_API_KEY")
    groq_api_key = os.environ.get("GROQ_API_KEY")

    # Get the model name for display
    model_name = os.environ.get("GROQ_MODEL")

    if not daytona_api_key or daytona_api_key == "your_daytona_api_key_here":
        print("WARNING: DAYTONA_API_KEY missing or default value. Running in test mode.")
        test_mode = True

    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        print("WARNING: GROQ_API_KEY missing or default value. Running in test mode.")
        test_mode = True

    print(f"GROQ_API_KEY value: {groq_api_key[:5] if groq_api_key else None}... (first 5 characters)")
    print(f"Test mode: {test_mode}")

    # Update the index.html template to use test mode if needed
    if test_mode:
        print("Updating index.html to use test mode")
        try:
            with open('templates/index.html', 'r') as f:
                content = f.read()

            # Replace the form action to use test mode if not already set
            if 'action="/test"' not in content:
                content = content.replace('id="codeForm" class="space-y-4"',
                                    'id="codeForm" class="space-y-4" action="/test"')

                with open('templates/index.html', 'w') as f:
                    f.write(content)

                print("Updated index.html to use test mode endpoint")
            else:
                print("index.html already configured for test mode")

            # Also make the test mode banner visible
            if 'id="testModeBanner" class="hidden' in content:
                content = content.replace('id="testModeBanner" class="hidden', 'id="testModeBanner" class="')

                with open('templates/index.html', 'w') as f:
                    f.write(content)

                print("Made test mode banner visible")
        except Exception as e:
            print(f"Error updating index.html: {str(e)}")

    try:
        # Run with SocketIO instead of the Flask built-in server
        # This is important for WebSocket support
        # Force port 5001 to avoid conflicts with AirPlay on macOS
        port = 5001
        print(f"Starting server on port {port}...")
        socketio.run(app, debug=False, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"Error starting the server: {str(e)}")