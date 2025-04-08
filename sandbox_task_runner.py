import os
import json
import sys
from groq import Groq

# Pre-set API key
os.environ["GROQ_API_KEY"] = "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"
print("DEBUG[sandbox]: Hard-coded GROQ_API_KEY is present:", bool(os.environ.get("GROQ_API_KEY")))

def generate_code(user_input):
    """Generate Python code based on user input using Groq API and LLaMA 4"""
    # Initialize Groq client
    groq_api_key = os.environ.get("GROQ_API_KEY")
    print(f"DEBUG[sandbox]: Using GROQ_API_KEY: {'Present' if groq_api_key else 'Missing'}")
    client = Groq(api_key=groq_api_key)
    
    # Generate code using LLaMA 4
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful AI assistant that generates Python code based on user requests. Generate ONLY the Python code without any explanations or markdown formatting."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    
    # Extract the generated code
    return chat_completion.choices[0].message.content

def execute_code(code):
    """Execute the generated code and capture output"""
    import sys
    from io import StringIO
    import traceback
    
    # First, strip markdown code blocks if present
    print(f"DEBUG[sandbox]: Original code length: {len(code)} chars")
    if code.startswith('```python') or code.startswith('```'):
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
            generated_code = generate_code(user_input)
            print(f"Successfully generated code for: {user_input}")
        except Exception as e:
            print(f"Error generating code: {str(e)}")
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
        
        # Return the results
        result = {
            "generated_code": generated_code,
            "execution_output": execution_results["execution_output"],
            "execution_result": execution_results["execution_result"]
        }
        
        result_file_path = os.path.join(home_dir, "result.json")
        with open(result_file_path, "w") as f:
            json.dump(result, f)
        
        return result
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"DEBUG[sandbox]: Process task error: {str(e)}\n{error_trace}")
        error_result = {
            "generated_code": f"Error occurred during processing: {str(e)}",
            "execution_output": error_trace,
            "execution_result": f"Error: {str(e)}"
        }
        import os
        home_dir = os.path.expanduser("~")
        error_file_path = os.path.join(home_dir, "error_result.json")
        with open(error_file_path, "w") as f:
            json.dump(error_result, f)
        return error_result

# The following code runs when this script is executed directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
    else:
        user_input = input("Enter your task: ")
    
    result = process_task(user_input)
    print(json.dumps(result))