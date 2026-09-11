import json
import math

def generate_new_math_samples():
    """
    Generates a list of new, perfectly formatted training samples
    that require the 'math' library.
    """
    new_samples = []

    # Problems involving math constants
    problems = {
        "Calculate the value of pi plus e.": "math.pi + math.e",
        "What is 2 times pi?": "2 * math.pi",
        "What is e squared?": "math.e**2",
    }

    # Problems involving basic math functions
    functions = {
        "sqrt": [144, 9801, 1000000],
        "log": [100, 1, 500],
        "log10": [1000, 10, 5000],
        "sin": [0, "math.pi/2", "math.pi"],
        "cos": [0, "math.pi/2", "math.pi"],
        "tan": ["math.pi/4", 0, 1],
        "factorial": [5, 10, 15],
        "degrees": ["math.pi", "math.pi/2", 3],
        "radians": [180, 90, 270],
    }

    # Generate problems for constants
    for p_text, p_code in problems.items():
        code = (
            f"import math\n\n"
            f"def solve():\n"
            f"    return {p_code}\n\n"
            f"print(solve())"
        )
        new_samples.append({
            "problem": p_text,
            "problem_type": "COMPUTATIONAL",
            "is_correct": True,
            "generated_solution": "...", # Placeholder, can be filled later
            "generated_python_code": code
        })

    # Generate problems for functions
    for func_name, values in functions.items():
        for val in values:
            problem_text = f"Calculate the value of math.{func_name}({val})"
            code = (
                f"import math\n\n"
                f"def solve():\n"
                f"    return math.{func_name}({val})\n\n"
                f"print(solve())"
            )
            new_samples.append({
                "problem": problem_text,
                "problem_type": "COMPUTATIONAL",
                "is_correct": True,
                "generated_solution": "...", # Placeholder
                "generated_python_code": code
            })
            
    # Add some more complex examples
    complex_problems = {
        "Calculate sqrt(16) + log10(100)": "math.sqrt(16) + math.log10(100)",
        "What is sin(pi/2) + cos(pi)?": "math.sin(math.pi/2) + math.cos(math.pi)",
        "Calculate the factorial of 6 plus the log of 20.": "math.factorial(6) + math.log(20)",
    }
    for p_text, p_code in complex_problems.items():
        code = (
            f"import math\n\n"
            f"def solve():\n"
            f"    return {p_code}\n\n"
            f"print(solve())"
        )
        new_samples.append({
            "problem": p_text,
            "problem_type": "COMPUTATIONAL",
            "is_correct": True,
            "generated_solution": "...", # Placeholder
            "generated_python_code": code
        })

    return new_samples

def augment_data(input_file, output_file):
    """
    Reads an existing JSONL file, adds new samples, and writes to a new file.
    """
    try:
        with open(input_file, 'r') as f:
            existing_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Input file {input_file} not found. Starting with new data.")
        existing_data = []

    new_samples = generate_new_math_samples()
    
    combined_data = existing_data + new_samples

    with open(output_file, 'w') as f:
        for entry in combined_data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Successfully augmented data.")
    print(f"Original samples: {len(existing_data)}")
    print(f"New samples added: {len(new_samples)}")
    print(f"Total samples in new file: {len(combined_data)}")
    print(f"New file saved to: {output_file}")

if __name__ == "__main__":
    input_path = "/local/user/gemini/arithmetic_logic_data.jsonl"
    output_path = "/local/user/gemini/arithmetic_logic_data_augmented.jsonl"
    augment_data(input_path, output_path)