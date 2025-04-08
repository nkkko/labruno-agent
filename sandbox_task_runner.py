import os
import json
import sys
from groq import Groq

def strip_markdown(code_text):
    """Remove markdown code block syntax from the generated code"""
    if code_text.startswith("```python"):
        # Find the end of the opening code block
        start_idx = code_text.find("\n") + 1
        # Find the closing code block
        end_idx = code_text.rfind("```")
        if end_idx > start_idx:
            return code_text[start_idx:end_idx].strip()
    
    # If no markdown formatting is detected or it's not formatted as expected
    # just return the original code but strip any triple backticks
    return code_text.replace("```", "").strip()

def generate_code(user_input):
    """Generate Python code based on user input using Groq API and LLaMA 4"""
    # Initialize Groq client
    groq_api_key = os.environ.get("GROQ_API_KEY")
    print(f"DEBUG[sandbox]: Using GROQ_API_KEY: {'Present' if groq_api_key else 'Missing'}")
    client = Groq(api_key=groq_api_key)
    
    # Generate code using LLaMA 4 with strong instructions to avoid markdown
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful AI assistant that generates Python code based on user requests. Generate ONLY the raw Python code without any explanations or markdown formatting. NEVER use ```python or any markdown formatting. Just output the plain Python code."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    
    # Extract the generated code and strip any markdown that might still be present
    code = chat_completion.choices[0].message.content
    cleaned_code = strip_markdown(code)
    print(f"DEBUG[sandbox]: Generated code length: {len(cleaned_code)} characters")
    return cleaned_code

def execute_code(code):
    """Execute the generated code and capture output"""
    import sys
    from io import StringIO
    import traceback
    
    # Modify the code to ensure it produces output
    # Check if it's a script with if __name__ == "__main__"
    if 'if __name__ == "__main__":' in code and 'main()' in code:
        # Add a call to main() at the end, regardless of the guard
        modified_code = code + "\n\n# Ensure main is called\ntry:\n    main()\nexcept NameError:\n    pass"
    else:
        # Check if the last line is an expression that may return a value
        lines = code.strip().split('\n')
        last_line = lines[-1].strip()
        
        # If the last line might be a function or expression that returns a value but doesn't print
        if (not last_line.startswith('print(') and 
            not last_line.startswith('#') and 
            not last_line.startswith('if ') and 
            not last_line.endswith(':') and
            not last_line == ''):
            # Modify the code to print the result of the last line
            lines[-1] = f"print('Result:', {last_line})"
            modified_code = '\n'.join(lines)
        else:
            modified_code = code
    
    print(f"DEBUG[sandbox]: Executing code (length: {len(modified_code)} chars)")
    
    output = StringIO()
    original_stdout = sys.stdout
    sys.stdout = output
    
    try:
        # Use a dedicated namespace to avoid conflicts
        exec_namespace = {}
        exec(modified_code, exec_namespace)
        execution_result = "Success"
    except Exception as e:
        error_msg = str(e)
        trace = traceback.format_exc()
        execution_result = f"Error: {error_msg}\n{trace}"
    
    sys.stdout = original_stdout
    execution_output = output.getvalue()
    
    # If there's still no output but execution was successful, add an info message
    if not execution_output.strip() and execution_result == "Success":
        execution_output = "(Code executed successfully but produced no output)"
    
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