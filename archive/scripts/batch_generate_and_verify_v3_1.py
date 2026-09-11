
#import tensorflow as tf
#print(tf.test.is_gpu_available())

import ollama
import pandas as pd
from datasets import Dataset
import time
import os
import json
import re
import sympy



# --- CONFIGURATION ---
INPUT_FILE = r"/mnt/ollama_data/ubuntu/projects/2025_0627/dataset/open_math_reasoning-mini-cot.arrow"
OUTPUT_FILE = r"/media/x/系统/gemini_files/open_math_reasoning_with_python_v4.arrow"
CHECKPOINT_FILE = r"/media/x/系统/gemini_files/checkpoint.json"
RESULTS_JSONL_FILE = r"/media/x/系统/gemini_files/results_v3.jsonl"
OLLAMA_MODEL = 'qwen-coder-32b-fp16:latest'
SAMPLES_TO_PROCESS = -1
BATCH_SIZE = 1
MAX_SOLUTION_LEN = 1300 * 2.4
# --- END CONFIGURATION ---

PROMPT_CLASSIFY = """
[SYSTEM]
You are a mathematical problem classifier. Your primary task is to determine if the ultimate goal of a problem is to find a specific numerical answer or to provide a general theoretical proof.

- **COMPUTATIONAL**: Choose this if the final goal is a number, a list of numbers, or a specific mathematical expression. The reasoning might involve deriving formulas or logical steps, but these steps are a means to an end: calculating a concrete result. This can be implemented in a Python function that returns a value.
- **PROOF**: Choose this if the final goal is to demonstrate the validity of a mathematical statement, theorem, or formula in a general case, without necessarily producing a specific numerical answer. The reasoning establishes a universal truth rather than a specific value.

[USER]
Problem:
{problem_text}

Reasoning:
{solution_cot_text}

Based on the ultimate goal of the problem and the provided reasoning, is it COMPUTATIONAL or PROOF?
Answer ONLY with one of the following words: COMPUTATIONAL, PROOF
"""

PROMPT_GENERATE_CODE = """
[SYSTEM]
You are an expert Python programmer. Your task is to translate a given mathematical reasoning process into a single, self-contained Python function.

[USER]
Based on the problem description and the provided step-by-step reasoning, write a Python function to solve the problem.

Problem:
{problem_text}

Reasoning:
{solution_cot_text}

Provide your answer strictly in the following format.

[PYTHON CODE]
```python
# Your self-contained Python code based on the reasoning above.
# The function should be named 'solve'.
# It should not take any arguments.
# It should return the final numerical answer. If there are multiple solutions, return a list of numbers.
```
"""

def is_numeric(s: str) -> bool:
    """Check if a string can be converted to a number (int or float)."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def classify_by_rules(problem: str, answer: str) -> str | None:
    """
    Applies a set of heuristics to quickly classify a problem.
    Returns a classification ('PROOF', 'COMPUTATIONAL', 'NON_COMPUTABLE') or None.
    """
    if 'infty' in problem or 'inf' in problem:
        return 'PROOF'
    if '\\sum' in problem and ('\\binom' in problem or '\\choose' in problem):
        limits = re.findall(r'_\{k=(\d+)\}\\^\{(\d+)\}', problem)
        for limit in limits:
            try:
                if int(limit[1]) > 100:
                    return 'PROOF'
            except (ValueError, IndexError):
                continue
    if (answer and '\\frac' in answer) or (answer and 'degree' in answer):
        return 'COMPUTATIONAL'
    if not isinstance(answer, str) or not answer.strip():
        return 'NON_COMPUTABLE'
    temp_answer = answer.lower().replace('pi', '').replace('e', '')
    if not any(char.isdigit() for char in temp_answer):
        return 'NON_COMPUTABLE'
    if len(answer) > 20 and not is_numeric(answer):
        return 'NON_COMPUTABLE'
    num_digits = sum(c.isdigit() for c in answer)
    total_len = len(answer)
    if total_len > 0 and (num_digits / total_len) < 0.15:
        return 'NON_COMPUTABLE'
    return None

def classify_problem_type(problem: str, solution_cot: str, expected_answer: str, model: str) -> str:
    rule_classification = classify_by_rules(problem, expected_answer)
    if rule_classification:
        return rule_classification
    formatted_prompt = PROMPT_CLASSIFY.format(problem_text=problem, solution_cot_text=solution_cot)
    raw_response = ollama_call_with_retry(model, formatted_prompt)
    if raw_response and "COMPUTATIONAL" in raw_response.strip().upper():
        return "COMPUTATIONAL"
    return "PROOF"

def ollama_call_with_retry(model: str, prompt: str, retries: int = 2, timeout: int = 70) -> str | None:
    # Instantiate a client with the desired timeout for this specific call.
    client = ollama.Client(timeout=timeout)
    for attempt in range(retries):
        try:
            response = client.generate(
                model=model,
                prompt=prompt,
                options={'temperature': 0.0},
                stream=False
            )
            return response['response']
        except Exception as e:
            print(f"  WARNING: Ollama call failed on attempt {attempt + 1}/{retries}. Error: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return None

def generate_python_code(problem: str, solution_cot: str, model: str) -> str | None:
    formatted_prompt = PROMPT_GENERATE_CODE.format(problem_text=problem, solution_cot_text=solution_cot)
    raw_response_text = ollama_call_with_retry(model, formatted_prompt)
    if raw_response_text and '[PYTHON CODE]' in raw_response_text:
        try:
            code_part = raw_response_text.split('[PYTHON CODE]')[1]
            python_code = code_part.replace('```python', '').replace('```', '').strip()
            return python_code if python_code else "None"
        except IndexError:
            return "None"
    return "None"

def execute_safely(code_string: str) -> tuple[any, str]:
    if not code_string or code_string == "None":
        return None, "No code to execute"
    try:
        scope = {}
        exec(code_string, scope)
        if 'solve' in scope and callable(scope['solve']):
            result = scope['solve']()
            return result, "Success"
        else:
            return None, "Execution failed: 'solve' function not found"
    except Exception as e:
        return None, f"Execution failed: {str(e)}"

def latex_to_expression(latex_str: str) -> str:
    """Converts a LaTeX string to a Python-evaluable expression."""
    # Remove delimiters
    latex_str = re.sub(r"^\s*\(?\s*(.*?)\s*\)?\s*$", r"\1", latex_str)
    latex_str = re.sub(r"^\s*\$\s*(.*?)\s*\$\s*$", r"\1", latex_str)
    
    # LaTeX command replacements
    replacements = {
        r"\\frac": r"(\1)/(\2)",
        r"\\cdot": r"*",
        r"\\times": r"*",
        r"\\pi": r"pi",
        r"\\^": r"**",
    }
    
    processed_str = latex_str
    for key, value in replacements.items():
        if key == r"\\frac":
            processed_str = re.sub(r"\\frac\{(.+?)\}\{(.+?)\}", r"(\1)/(\2)", processed_str)
        elif key == r"\\^":
            processed_str = re.sub(r"\\^\{(.+?)\}", r"**(\1)", processed_str)
            processed_str = re.sub(r"\\^([\\d\\w.]+)", r"**\\1", processed_str)
        else:
            processed_str = processed_str.replace(key, value)
            
    return processed_str


def compare_answers(result, expected) -> bool | None:
    if result is None or expected is None:
        return None

    try:
        import sympy
        import re
    except ImportError:
        return False

    result_str = str(result).strip()
    expected_str = str(expected).strip()

    if result_str == expected_str:
        return True

    try:
        # --- LaTeX to Sympy String Conversion ---
        # Remove LaTeX delimiters
        if expected_str.startswith('\\(') and expected_str.endswith('\\)'):
            expected_str = expected_str[2:-2]
        elif expected_str.startswith('$') and expected_str.endswith('$'):
            expected_str = expected_str[1:-1]

        # A more robust replacement strategy
        expected_str = re.sub(r'\\frac{([^}]+)}{([^}]+)}', r'((\1))/(\2)', expected_str)
        expected_str = re.sub(r'\\sqrt{([^}]+)}', r'sqrt(\1)', expected_str)
        expected_str = re.sub(r'\\sqrt(.)', r'sqrt(\1)', expected_str)
        expected_str = expected_str.replace('\\pi', 'pi')
        expected_str = expected_str.replace('^', '**')
        expected_str = expected_str.replace('\\cdot', '*').replace('\\times', '*')
        expected_str = expected_str.replace('{', '(').replace('}', ')')
        # --- End Conversion ---

        # Add a local dict for safe sympify, including e
        local_dict = {'sqrt': sympy.sqrt, 'pi': sympy.pi, 'e': sympy.E}
        result_expr = sympy.sympify(result_str, locals=local_dict)
        expected_expr = sympy.sympify(expected_str, locals=local_dict)

        # Evaluate numerically and compare with tolerance
        result_val = result_expr.evalf(30)
        expected_val = expected_expr.evalf(30)

        # Explicitly convert the sympy boolean to a python boolean
        return bool(sympy.Abs(result_val - expected_val) < 1e-9)

    except Exception:
        return False



def save_checkpoint(index: int):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump({'last_processed_index': index}, f)

def load_checkpoint() -> int:
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if not isinstance(data, dict): return 0
            return data.get('last_processed_index', 0)
        except json.JSONDecodeError:
            return 0


def test_compare_answers():
    """A dedicated function to test the compare_answers logic."""
    print("--- Running tests for compare_answers ---")
    test_cases = [
        (847288609443, "\\(3^{25}\\)", True),
        (4.810477380965351, "\\( e^{\\frac{\\pi}{2}} \\)", True),
        (0.6666666666666666, "\\(\\frac{2}{3}\\)", True),
        (100, "100", True),
        (0.01, "\\(\\frac{1}{100}\\)", True),
        (3.141592653589793, "\\pi", True),
        ("1/2", "0.5", True),
        (7.0710678118654755, "\\(\\sqrt{50}\\)", True),
        ("1.0e-3", "0.001", True),
        ("invalid_result", "10", False),
        (10, "invalid_expected", False),
        (0.36787944117, "e^-1", True)
    ]

    all_passed = True
    for i, (result, expected, expected_outcome) in enumerate(test_cases):
        actual_outcome = compare_answers(result, expected)
        passed = actual_outcome == expected_outcome
        print(f"Test {i+1}: result='{result}', expected='{expected}'")
        print(f"  -> Expected: {expected_outcome}, Got: {actual_outcome} ... {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False

    print("--- Test summary ---")
    if all_passed:
        print("All tests passed successfully!")
    else:
        print("Some tests failed.")
    print("--------------------\\n")



def main():
    # Run the dedicated test function first to verify its reliability
    #test_compare_answers()

    #print("Test complete. To run the full script, remove the call to 'test_compare_answers()' from main().")

    print("Starting robust batch processing v4 (Rules + LLM)...")
    
    try:
        full_df = Dataset.from_file(INPUT_FILE)
        print(f"Successfully loaded {len(full_df)} samples.")
    except Exception as e:
        print(f"FATAL: Could not read Arrow dataset. Error: {e}"); return

    start_index = load_checkpoint()
    if start_index > 0:
        print(f"Resuming from saved checkpoint at index {start_index}.")
        if not os.path.exists(RESULTS_JSONL_FILE):
            print("WARNING: Checkpoint found, but results.jsonl is missing. Starting from scratch.")
            start_index = 0
            
    if start_index == 0 and os.path.exists(RESULTS_JSONL_FILE):
        os.remove(RESULTS_JSONL_FILE)

    end_index = len(full_df) if SAMPLES_TO_PROCESS == -1 else min(SAMPLES_TO_PROCESS, len(full_df))

    with open(RESULTS_JSONL_FILE, 'a', encoding='utf-8') as results_file:
        for i in range(start_index, end_index):
            try:
                print(f"--- Processing sample {i+1}/{end_index} ---")
                
                row = full_df[i]
                problem, solution_cot, expected_answer = row['problem'], row['generated_solution'], row['expected_answer']
                
                if len(solution_cot) > MAX_SOLUTION_LEN:
                    print("The length of sample surpass the MAX_SOLUTION_LEN {} as {}".format(MAX_SOLUTION_LEN, len(solution_cot)))
                    continue
                print("  - Classifying problem...")
                problem_type = classify_problem_type(problem, solution_cot, expected_answer, OLLAMA_MODEL)
                print(f"  - Classification result: {problem_type}")
                
                code, result, status, is_correct = "N/A", "N/A", "N/A", None

                if problem_type == "COMPUTATIONAL":
                    print("  - Generating Python code...")
                    code = generate_python_code(problem, solution_cot, OLLAMA_MODEL)
                    
                    print("  - Executing generated code...")
                    result, status = execute_safely(code)
                    print(f"  - Execution status: {status}")
                    print(f"  - Execution result: {result}")

                    print("  - Comparing answers...")
                    is_correct = compare_answers(result, expected_answer)
                    print(f"  - Comparison result: {'Correct' if is_correct else 'Incorrect' if is_correct is not None else 'N/A'}")

                else:
                    status = f"Skipped: Classified as {problem_type}"
                    print(f"  - {status}")

                record = {
                    'problem': problem,
                    'generated_solution': solution_cot,
                    'expected_answer': expected_answer,
                    'problem_type': problem_type,
                    'generated_python_code': code,
                    'execution_result': str(result) if status == "Success" else status,
                    'is_correct': is_correct
                }
                
                results_file.write(json.dumps(record) + '\n')

            except Exception as e:
                print(f"  ERROR: An unexpected error occurred while processing sample {i+1}. Skipping.")
                print(f"  Error details: {e}")
                error_record = {
                    'problem': full_df[i].get('problem', 'Unknown'),
                    'error_message': str(e)
                }
                continue

            if (i + 1) % BATCH_SIZE == 0 or (i + 1) == end_index:
                results_file.flush()
                save_checkpoint(i + 1)
                print(f"--- Checkpoint saved at index {i + 1}. ---\n")
                #To take a break to cool down machine
                time.sleep(6)
            
            time.sleep(0.5)

    print("\n--- Incremental Processing Complete ---")
    print(f"All results saved incrementally to {RESULTS_JSONL_FILE}")

    print("Converting final results to Arrow format...")
    try:
        final_df = pd.read_json(RESULTS_JSONL_FILE, lines=True)
        final_dataset = Dataset.from_pandas(final_df)
        final_dataset.to_feather(OUTPUT_FILE)
        print(f"\nSuccessfully converted and saved final data to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nFATAL: Could not convert or save final Arrow file. Error: {e}")
        print(f"Your intermediate results are safe in: {RESULTS_JSONL_FILE}")

if __name__ == '__main__':
    main()
