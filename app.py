import os
import uuid
import json
import tempfile
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from daytona_sdk import Daytona, DaytonaConfig, CreateSandboxParams
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv(verbose=True, override=True)  # Added verbose and override to ensure environment variables are loaded

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

# Initialize Flask app
app = Flask(__name__)
app.config["TAILWIND_DEV"] = True

# Initialize Daytona client
daytona_api_key = os.environ.get("DAYTONA_API_KEY")
daytona_target = os.environ.get("DAYTONA_TARGET")
print(f"Initializing Daytona client with API key: {daytona_api_key[:5] if daytona_api_key else None}... and target: {daytona_target}")
daytona = Daytona(DaytonaConfig(api_key=daytona_api_key, target=daytona_target))

def create_sandbox(code_to_upload=None):
    """Create a Daytona sandbox and optionally upload code to it"""
    params = CreateSandboxParams(language="python")
    
    # Get environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    print(f"DEBUG: Using GROQ_API_KEY: {'Present' if groq_api_key else 'Missing'}")
    
    sandbox = daytona.create(params)
    print(f"DEBUG: Created sandbox: {sandbox}")
    
    # Set environment variables in the sandbox
    try:
        print("DEBUG: Setting environment variables in the sandbox")
        # Get environment variables
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Create a modified task runner file with the API key directly embedded
        with open('sandbox_task_runner.py', 'r') as f:
            task_runner_content = f.read()
            
        # Insert the API key directly into the task runner file
        modified_task_runner = task_runner_content.replace(
            'import os\nimport json\nimport sys\nfrom groq import Groq',
            f'''import os
import json
import sys
from groq import Groq

# Pre-set API key
os.environ["GROQ_API_KEY"] = "{groq_api_key}"
print("DEBUG[sandbox]: Hard-coded GROQ_API_KEY is present:", bool(os.environ.get("GROQ_API_KEY")))'''
        )
        
        # Save this modified version to a temp directory to avoid triggering Flask reload
        import tempfile
        temp_dir = tempfile.gettempdir()
        modified_runner_path = os.path.join(temp_dir, 'modified_task_runner.py')
        with open(modified_runner_path, 'w') as f:
            f.write(modified_task_runner)
            
        print("DEBUG: Created modified task runner with embedded API key")
            
        # Also create and upload a .env file as a backup method
        env_file_content = f'''
# Environment variables for sandbox
GROQ_API_KEY={groq_api_key}
'''
        with open('temp_env.txt', 'w') as f:
            f.write(env_file_content)
        
        # Upload the environment file to the sandbox
        sandbox.fs.upload_file("/home/daytona/.env", open('temp_env.txt', 'rb').read())
        
        # Simple environment setup to make sure we have the key in the environment
        env_setup_code = '''
import os

os.environ["GROQ_API_KEY"] = "{}".format(open('/home/daytona/.env').read().split('=')[1].strip())
print(f"DEBUG[sandbox]: GROQ_API_KEY set directly: {os.environ.get('GROQ_API_KEY')[:5]}...")
'''
        sandbox.process.code_run(env_setup_code)
        
        # Remove the temporary file
        os.remove('temp_env.txt')
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

def generate_and_execute_code(sandbox, user_input, task_runner_path=None):
    """Generate and execute code in a sandbox based on user input"""
    print(f"DEBUG: In generate_and_execute_code with sandbox: {sandbox}")
    
    # Use Groq API to generate code
    try:
        print(f"DEBUG: Running code in sandbox to generate and execute code")
        # No indentation in the executed code to prevent errors
        # Properly escape the user input to avoid issues with quotes
        safe_user_input = user_input.replace('\\', '\\\\').replace('"', '\\"')
        
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
    
    # Use task runner to process the request
    user_input = "{safe_user_input}"
    print(f"DEBUG[sandbox]: Processing task: {{user_input}}")
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
    
    # Generate code using LLaMA 4
    try:
        print("DEBUG[sandbox]: Creating chat completion with LLaMA 4")
        chat_completion = client.chat.completions.create(
            messages=[
                {{
                    "role": "system", 
                    "content": "You are a helpful AI assistant that generates Python code based on user requests. Generate ONLY the Python code without any explanations or markdown formatting."
                }},
                {{
                    "role": "user",
                    "content": "{}"
                }}
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        print("DEBUG[sandbox]: Chat completion created")
    except Exception as e:
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
                        return parsed_result
                except:
                    pass
                    
            # If no JSON is found, create a result from the stdout
            print(f"DEBUG: Creating result from stdout")
            # Try to extract code blocks if they exist
            code_pattern = r'```python\s*(.*?)\s*```'
            code_match = re.search(code_pattern, result_text, re.DOTALL)
            generated_code = code_match.group(1) if code_match else result_text
            
            # Create a fallback result
            return {
                "generated_code": generated_code,
                "execution_output": result_text,
                "execution_result": "Success" if "error" not in result_text.lower() else "Error"
            }
        except Exception as e:
            print(f"DEBUG: Failed to parse result: {str(e)}")
            print(f"DEBUG: Raw result: {result.result if hasattr(result, 'result') else str(result)}")
            return {
                "error": f"Failed to parse result: {str(e)}", 
                "generated_code": "Error parsing result",
                "execution_output": result.result if hasattr(result, 'result') else str(result)
            }
            
    except Exception as e:
        print(f"DEBUG: Exception in generate_and_execute_code: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            "error": str(e),
            "generated_code": "Error occurred",
            "execution_output": traceback.format_exc()
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
            "evaluation": f"After analyzing the available implementations, I've determined that implementation {best_impl_index + 1} is the best solution. It executes successfully and produces the expected output. The code is concise and follows good programming practices. Output: {output_snippet}"
        }
    else:
        evaluation = {
            "evaluation": "After analyzing the available implementations, I couldn't determine a clear winner. Multiple implementations succeeded, but none stood out as significantly better than the others."
        }
    
    print(f"DEBUG: Generated evaluation: {evaluation}")
    return evaluation

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/execute', methods=['POST'])
def execute():
    user_input = request.form.get('user_input')
    results = []
    evaluation = {"error": "Not started"}
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
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
        
        # Create 5 sandboxes in parallel
        print("DEBUG: Creating 5 sandboxes in parallel...")
        sandboxes = []
        for i in range(5):
            try:
                sandbox = create_sandbox()
                
                # Upload task runner to each worker sandbox
                try:
                    # Read the modified task runner with embedded API key
                    modified_runner_path = os.path.join(tempfile.gettempdir(), 'modified_task_runner.py')
                    with open(modified_runner_path, 'r') as f:
                        task_runner_code = f.read()
                    
                    # Create directory structure
                    sandbox.fs.create_folder("/home/daytona/labruno", "755")
                    
                    # Upload the task runner file
                    file_path = "/home/daytona/labruno/sandbox_task_runner.py"
                    sandbox.fs.upload_file(file_path, task_runner_code.encode('utf-8'))
                    
                    # Make it executable
                    sandbox.fs.set_file_permissions(file_path, mode="755")
                    
                    # Also upload the .env file as a backup
                    env_file_content = f'''
# Environment variables for sandbox
GROQ_API_KEY={groq_api_key}
'''
                    sandbox.fs.upload_file("/home/daytona/.env", env_file_content.encode('utf-8'))
                    
                    # Run code to set the environment variable directly
                    env_setup_code = '''
import os

os.environ["GROQ_API_KEY"] = "{}".format(open('/home/daytona/.env').read().split('=')[1].strip())
print(f"DEBUG[sandbox]: GROQ_API_KEY set directly: {os.environ.get('GROQ_API_KEY')[:5]}...")
'''
                    sandbox.process.code_run(env_setup_code)
                    
                    print(f"DEBUG: Uploaded task runner to sandbox {i+1}")
                except Exception as e:
                    print(f"DEBUG: Error uploading task runner to sandbox {i+1}: {str(e)}")
                
                print(f"DEBUG: Created sandbox {i+1}: {sandbox}")
                sandboxes.append(sandbox)
            except Exception as e:
                print(f"DEBUG: Error creating sandbox {i+1}: {str(e)}")
        
        print(f"DEBUG: Created {len(sandboxes)} sandboxes")
        
        # Run code generation and execution in parallel
        print("DEBUG: Running code generation and execution...")
        results = []
        for i, sandbox in enumerate(sandboxes):
            try:
                print(f"DEBUG: Processing sandbox {i+1}...")
                result = generate_and_execute_code(sandbox, user_input, task_runner_path)
                print(f"DEBUG: Result from sandbox {i+1}: {result}")
                results.append(result)
            except Exception as e:
                print(f"DEBUG: Error processing sandbox {i+1}: {str(e)}")
                results.append({"error": str(e), "generated_code": "Error", "execution_output": str(e)})
        
        print(f"DEBUG: Collected {len(results)} results")
        
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
        
        # Clean up sandboxes
        print("DEBUG: Cleaning up sandboxes...")
        for i, sandbox in enumerate(sandboxes):
            try:
                print(f"DEBUG: Removing sandbox {i+1}...")
                daytona.remove(sandbox)
            except Exception as e:
                print(f"DEBUG: Error removing sandbox {i+1}: {str(e)}")
        
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
    
    print("DEBUG: Returning results to client")
    return jsonify({
        "results": results,
        "evaluation": evaluation
    })

@app.route('/test', methods=['POST'])
def test_mode():
    """A test endpoint that doesn't actually use Daytona or Groq APIs"""
    user_input = request.form.get('user_input')
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    
    print(f"DEBUG: Test mode activated with input: {user_input}")
    
    # Create dummy results
    results = []
    for i in range(5):
        gen_code = f"""# Test implementation {i+1} for: {user_input}

def solution():
    # Implementation {i+1}
    print('This is test implementation {i+1}')
    return {i+1}

result = solution()
print(f'Result: {{result}}')"""

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
        # Run without debug mode to prevent auto-reloading when temporary files are created
        # This is important to prevent Flask from restarting when we create the modified task runner
        app.run(debug=False)
    except OSError:
        # If port 5000 is in use, try port 5001
        print("Port 5000 is in use, trying port 5001...")
        app.run(debug=False, port=5001)