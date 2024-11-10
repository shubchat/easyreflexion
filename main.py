from human_eval.data import write_jsonl, read_problems
from openai import AzureOpenAI

# Initialize the Azure OpenAI client or if you want you can replace it with direct openAI/Claude API end point
client = AzureOpenAI(
    azure_endpoint = "", 
    api_key="",  
    api_version=""
)



def prepare_for_llm(content, max_length=10000):
    """
    Prepare content for LLM by ensuring it doesn't exceed context window.
    """
    if len(content) <= max_length:
        return content
        
    # For test results/analysis content
    if isinstance(content, list):
        return content[:3]  # Keep only first 3 test results if too many
        
    # For code and history content
    if "```" in content:
        # Preserve code blocks but limit their size
        parts = content.split("```")
        processed_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Code block
                if len(part) > max_length // 2:
                    processed_parts.append(part[:max_length//2])
                else:
                    processed_parts.append(part)
            else:  # Regular text
                if len(part) > max_length // 4:
                    processed_parts.append(part[:max_length//4])
                else:
                    processed_parts.append(part)
        return "```".join(processed_parts)

    # For history text
    if isinstance(content, str) and "Attempt" in content:
        attempts = content.split("Attempt")
        # Keep most recent attempts
        return "Attempt".join([""] + attempts[-2:])

    # Default truncation
    return (content[:max_length//2] + 
            "\n...[truncated]...\n" + 
            content[-max_length//2:])


def generate_one_completion(prompt):
    """
    Generate solution through test case creation, implementation, and reflection.
    """
    MAX_ITERATIONS = 3

    # Step 1: Generate test cases
    print("Generating test cases...")
    test_case_response = client.chat.completions.create(
        model="mendi-ml",
        messages=[
            {"role": "system", "content": """Analyze the problem and create comprehensive test cases.
            Consider:
            1. Basic functionality cases
            2. Edge cases (empty, null, boundaries)
            3. Special cases
            4. Invalid inputs
            
            Format your response exactly like this for each test case:
        TEST CASE 1:
        Input: <provide exact input that can be evaluated>
        Expected: <provide exact expected output>
        Explanation: <explain why this test case is important>

        TEST CASE 2:
        Input: <provide exact input that can be evaluated>
        Expected: <provide exact expected output>
        Explanation: <explain why this test case is important>
        
        Example format:
        TEST CASE 1:
        Input: [1, 2, 3]
        Expected: 6
        Explanation: Tests basic functionality with positive integers

        TEST CASE 2:
        Input: []
        Expected: 0
        Explanation: Edge case - empty list should return 0

        Make sure all inputs and expected outputs are in valid Python syntax that can be evaluated"""},
            {"role": "user", "content": prepare_for_llm(prompt)}
        ],
        temperature=0.5,
        max_tokens=1000
    )
    test_cases = test_case_response.choices[0].message.content

    # Step 2: Generate initial solution
    print("Generating initial solution...")
    code_response = client.chat.completions.create(
        model="mendi-ml",
        messages=[
            {"role": "system", "content": """You are an expert Python programmer. Provide only the function implementation without any explanations. In the function also include any imports if needed,Break down the problem into smaller functions or components:

    Analyze the problem requirements
    Identify key functions or components needed
    Outline the main function structure
    Implement helper functions
    Combine components in the main function"""},
            {"role": "user", "content": f"Problem:{prepare_for_llm(prompt)}"}
        ],
        temperature=0.5,
        max_tokens=1000
    )
    current_code = code_response.choices[0].message.content.replace('```python', '').replace('```', '').strip()
   
    def print_test_summary(results):
        """Print a summary of test results."""
        passed_count = sum(1 for result in results if result['passed'])
        failed_count = len(results) - passed_count
        
        print("\n=== Test Summary ===")
        print(f"Total Tests: {len(results)}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        
        if failed_count > 0:
            print("\nFailed Tests Details:")
            for result in results:
                if not result['passed']:
                    print(f"Input: {result['input']}")
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"Expected: {result['expected']}, Actual: {result['actual']}")
                    print(f"Explanation: {result['explanation']}\n")
        # Step 3: Test and reflect loop
    for iteration in range(MAX_ITERATIONS):
        print(f"\nIteration {iteration + 1}/{MAX_ITERATIONS}")
        
        # Execute tests
        results = execute_tests(current_code, test_cases)
        print_test_summary(results)
        
        # print(results)
        print("Converting test results to natural language...")
        test_analysis_response = client.chat.completions.create(
            model="mendi-ml",
            messages=[
                {"role": "system", "content": """Analyze these test results and explain them in clear natural language.
                For each test case:
                - Explain what was tested
                - Whether it passed or failed
                - If failed, explain why it failed and what the discrepancy was
                Keep your explanation technical,clear and short."""},
                {"role": "user", "content": f"Here are the test results:\n{prepare_for_llm(results)}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        test_analysis = test_analysis_response.choices[0].message.content
        if iteration == 0:
            improvement_history = []
        improvement_history.append({
            'iteration': iteration + 1,
            'code': current_code,
            'analysis': test_analysis
        })

        # Format history for reflection
        history_text = "\n\n".join([
            f"Attempt {h['iteration']}:\n"
            f"Code:\n{h['code']}\n"
            f"Analysis:\n{h['analysis']}"
            for h in improvement_history
        ])
        history_text = prepare_for_llm(history_text)
        if all(r['passed'] for r in results):
            print("All tests passed!")
            return current_code

        # If tests failed, generate improved version
        print("Generating improved solution...")
        reflection_response = client.chat.completions.create(
            model="mendi-ml",
            messages=[
                {"role": "system", "content": "You are improving code based on test results."},
                {"role": "user", "content": f"""
                                                Previous attempts and their analyses:
                    {history_text}

                    Current code:
                    {prepare_for_llm(current_code)}

                    Latest test analysis:
                    {prepare_for_llm(test_analysis)}


                                                Based on this analysis, provide an improved implementation that fixes these issues.
                                                Provide only the code."""}
                                                    ],
                                                    temperature=0.5,
                                                    max_tokens=1000
                                                )
        current_code = reflection_response.choices[0].message.content.replace('```python', '').replace('```', '').strip()

    return current_code

# Helper function to execute tests
import re

def get_function_name(code):
    """Extract the function name from the code."""
    # Look for 'def function_name(' pattern
    match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
    if match:
        return match.group(1)
    return None

def execute_tests(code, test_cases):
    function_name = get_function_name(code)
    if not function_name:
        raise ValueError("Could not find function name in code")     
    print(f"Found function name: {function_name}")  # Debug print
    try:       
       namespace = {}
       exec(code, namespace)
       parsed_tests = parse_test_cases(test_cases, namespace)
       
       results = []
       for test in parsed_tests:
           try:
               test_input = test['input']
               expected = test['expected']
               explanation = test['explanation']
               
               # Evaluate the input
               evaluated_input = eval(test_input, namespace)
               
               # Handle different input types
               if isinstance(evaluated_input, (list, tuple)):
                   # If input is a list/tuple but function expects multiple args
                   try:
                       actual = namespace[function_name](*evaluated_input)
                   except TypeError:
                       # If unpacking failed, try passing as single argument
                       actual = namespace[function_name](evaluated_input)
               else:
                   # For non-sequence inputs, pass directly
                   actual = namespace[function_name](evaluated_input)
               
               passed = actual == expected
               
               results.append({
                   'passed': passed,
                   'input': test_input,
                   'expected': expected, 
                   'actual': actual,
                   'explanation': explanation
               })
               
           except Exception as e:
               results.append({
                   'passed': False,
                   'input': test_input,
                   'error': str(e),
                   'explanation': explanation
               })
               
       return results
       
    except Exception as e:
       return [{
           'passed': False,
           'error': f"Failed to execute code: {str(e)}"
       }]

def parse_test_cases(test_cases_str, namespace=None):
   """
   Parse test cases from LLM output string into structured format.
   
   Args:
       test_cases_str (str): Raw test cases string from LLM
       namespace (dict): Namespace for evaluating expressions
       
   Returns:
       list: List of dictionaries containing parsed test cases
   """
   parsed = []
   
   try:
       # Create empty namespace if none provided
       if namespace is None:
           namespace = {}
       
       # Split test cases by line and parse
       lines = test_cases_str.strip().split('\n')
       current_test = {}
       
       for line in lines:
           line = line.strip()
           if line.startswith('Input:'):
               if current_test:
                   parsed.append(current_test)
               current_test = {'input': line.replace('Input:', '').strip()}
           elif line.startswith('Expected:'):
               # Use namespace when evaluating expected output
               current_test['expected'] = eval(line.replace('Expected:', '').strip(), namespace)
           elif line.startswith('Explanation:'):
               current_test['explanation'] = line.replace('Explanation:', '').strip()
               
       # Add last test case
       if current_test:
           parsed.append(current_test)
           
       return parsed
       
   except Exception as e:
       print(f"Error parsing test cases: {e}")
       return []

# Main execution
print("Reading problems from source...")
problems = read_problems()
print("Problems loaded successfully.")

import signal
import time
from functools import wraps
from typing import Any, Callable
import random

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

class TimeoutError(Exception):
    pass

def with_timeout(seconds: int, default: Any = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Set the signal handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Disable the alarm
                return result
            except TimeoutError:
                signal.alarm(0)  # Disable the alarm
                return default
            finally:
                signal.alarm(0)  # Ensure alarm is disabled
        return wrapper
    return decorator

def generate_one_completion_with_retry(prompt, max_retries=3, timeout_seconds=300):
    """
    Wrapper function that handles timeouts and retries for generate_one_completion
    """
    @with_timeout(timeout_seconds)
    def attempt_generation():
        return generate_one_completion(prompt)

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}/{max_retries}")
            result = attempt_generation()
            if result is not None:
                return result
            
            print(f"Attempt {attempt + 1} timed out after {timeout_seconds} seconds")
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {str(e)}")
        
        # If we haven't succeeded, wait before retrying
        if attempt < max_retries - 1:
            backoff_time = random.uniform(1, 5) * (attempt + 1)
            print(f"Waiting {backoff_time:.2f} seconds before retrying...")
            time.sleep(backoff_time)
    
    raise Exception(f"Failed to generate completion after {max_retries} attempts")

# Modified main execution
print("Reading problems from source...")
problems = read_problems()
print("Problems loaded successfully.")

samples = []
for task_id in problems:
    numeric_id = int(task_id.split("/")[1])
    if numeric_id > 122:
        try:
            completion = generate_one_completion_with_retry(
                problems[task_id]["prompt"],
                max_retries=3,
                timeout_seconds=300  # 5 minutes timeout
            )
            samples.append(dict(task_id=task_id, completion=completion))
            print(f"✅ Completed task {task_id}")
        except Exception as e:
            print(f"❌ Failed task {task_id}: {str(e)}")
            # Add failed attempt to samples with error message
            samples.append(dict(
                task_id=task_id,
                completion=None,
                error=str(e)
            ))
        
        print("Writing intermediate samples to file...")
        write_jsonl("samples_reflexion_intermediate2.jsonl", samples)

    print("Writing final samples to file...")
    write_jsonl("samples_reflexion.jsonl", samples)

