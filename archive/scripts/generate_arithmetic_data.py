import json
import random

# --- Configuration ---
OUTPUT_FILE = "/local/user/gemini/arithmetic_logic_data.jsonl"
NUM_SAMPLES = 200  # Total number of problems to generate
# Define the operations and their corresponding Python syntax
OPERATIONS = {
    # Arithmetic
    'addition': ('+', "Calculate the sum of {} and {}."),
    'subtraction': ('-', "Calculate the difference between {} and {}."),
    'multiplication': ('*', "Calculate the product of {} and {}."),
    'integer division': ('//', "What is the result of the integer division of {} by {}?"),
    # Bitwise
    'bitwise NOT': ('~', "What is the bitwise NOT of {}?"),
    'bitwise AND': ('&', "What is the bitwise AND of {} and {}?"),
    'bitwise OR': ('|', "What is the bitwise OR of {} and {}?"),
    'bitwise XOR': ('^', "What is the bitwise XOR of {} and {}?"),
    'left shift': ('<<', "What is {} left-shifted by {} bits?"),
    'right shift': ('>>', "What is {} right-shifted by {} bits?"),
}
# Add 3-operand variations for arithmetic
THREE_OPERAND_OPS = {
    'addition': ('+', "Calculate the sum of {}, {}, and {}."),
    'multiplication': ('*', "Calculate the product of {}, {}, and {}."),
}

def generate_number(bits=64):
    """Generates a random integer with a specified number of bits."""
    return random.randint(2**(bits-1), 2**bits - 1)

def generate_shift_amount():
    """Generates a reasonable shift amount (1-63)."""
    return random.randint(1, 63)

# --- Data Generation ---
generated_records = []

print(f"Generating {NUM_SAMPLES} arithmetic and bitwise logic problems...")

for i in range(NUM_SAMPLES):
    use_three_operands = random.choice([True, False]) and i < (NUM_SAMPLES / 4)    # To keep amount under one fourth
    expression = ""
    
    if use_three_operands:
        op_name, (op_symbol, template) = random.choice(list(THREE_OPERAND_OPS.items()))
        n1 = generate_number(bits=random.choice([16, 64]))
        n2 = generate_number(bits=random.choice([16, 64]))
        n3 = generate_number(bits=random.choice([16, 64]))
        
        problem_text = template.format(n1, n2, n3)
        expression = f"{n1} {op_symbol} {n2} {op_symbol} {n3}"
        
    else:  # One or Two operands
        op_name, (op_symbol, template) = random.choice(list(OPERATIONS.items()))
        
        if op_name == 'bitwise NOT':
            n1 = generate_number(bits=random.choice([16, 64]))
            problem_text = template.format(n1)
            expression = f"{op_symbol}{n1}"
        
        elif 'shift' in op_name:
            n1 = generate_number(bits=64)
            n2 = generate_shift_amount()
            problem_text = template.format(n1, n2)
            expression = f"{n1} {op_symbol} {n2}"
            
        elif op_name == 'integer division':
            n1 = generate_number(bits=64)
            n2 = generate_number(bits=32)
            problem_text = template.format(n1, n2)
            expression = f"{n1} {op_symbol} {n2}"

        else:  # Standard binary operations
            n1 = generate_number(bits=random.choice([16, 64]))
            n2 = generate_number(bits=random.choice([16, 64]))
            problem_text = template.format(n1, n2)
            expression = f"{n1} {op_symbol} {n2}"

    # --- Format the code with the solve() function ---
    code_solution = (
        f"def solve():\n"
        f"    # It should not take any arguments.\n"
        f"    # It should return the final numerical answer.\n"
        f"    return {expression}\n\n"
        f"print(solve())"
    )
    
    # Calculate the answer for the mock solution text
    actual_answer = eval(expression)

    # Create the final record
    record = {
        "problem": problem_text,
        "problem_type": "COMPUTATIONAL",
        "is_correct": True,
        "generated_solution": f"The answer is {actual_answer}.",
        "generated_python_code": code_solution
    }
    generated_records.append(record)

# --- Write to File ---
try:
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for record in generated_records:
            f.write(json.dumps(record) + '\n')
    print(f"\nSuccessfully generated {len(generated_records)} samples.")
    print(f"Data saved to {OUTPUT_FILE}")
except Exception as e:
    print(f"An error occurred while writing to the file: {e}")
