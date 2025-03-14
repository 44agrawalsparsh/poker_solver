from enum import Enum
import os
import pdb
import subprocess
import json

class Position(Enum):
    UTG = 2
    HJ = 3
    CO = 4
    BTN = 5
    SB = 0
    BB = 1

class GAME_CODES(Enum):
    ALIVE = 0
    TOO_MANY_POSTFLOP = 1
    FINISHED = 2

PREFLOP_STRATEGY_PATH = "ranges/processed_preflop.json"
PREFLOP_RANGES_PATH = "ranges/preflop_ranges"
SOLVER_SCRIPT = "./solve_script.sh"
SOLVER_OUTPUT_PATH = "solver_output.json"

SOLVER_IP_RANGE_FILE = "solver_input_files/ranges/ip_range.txt"
SOLVER_OOP_RANGE_FILE = "solver_input_files/ranges/oop_range.txt"
SOLVER_INPUT_FILE = "solver_input_files/input.txt"


def find_existing_file(filepath):
    """
    Searches for the file specified by filepath. If it doesn't exist in its original location,
    the function will move up the directory tree, looking for a file with the same basename.
    
    :param filepath: The file path, including the filename.
    :return: The full path where the file was found, or None if it wasn't found.
    """
    filepath = os.path.abspath(filepath)
    # If the file exists at the given path, return it.
    if os.path.isfile(filepath):
        return filepath

    # Extract the filename and set the starting directory to the file's directory.
    basename = os.path.basename(filepath)
    current_dir = os.path.dirname(filepath)

    while True:
        candidate = os.path.join(current_dir, basename)
        if os.path.isfile(candidate):
            return candidate
        parent_dir = os.path.dirname(current_dir)
        # If we've reached the root directory (no more parent), return None.
        if parent_dir == current_dir:
            return None
        current_dir = parent_dir

def solve_postflop(oop_range, ip_range, board, pot, effective_stacks):
    board = ' '.join(board)

    #effective_stacks = min(8*pot, effective_stacks) ###lowkey very unideal to do but makes the game more solvable and should be fine for most cases

    with open(SOLVER_IP_RANGE_FILE, 'w') as f:
        f.write(ip_range)
    
    with open(SOLVER_OOP_RANGE_FILE, 'w') as f:
        f.write(oop_range)

    # Open the file and read its lines
    with open(SOLVER_INPUT_FILE, "r") as file:
        lines = file.readlines()

    # Ensure the file has at least 5 lines before overwriting
    if len(lines) < 5:
        raise ValueError("The file does not have at least 5 lines to replace.")

    # Replace the 3rd, 4th, and 5th lines (indices 2, 3, and 4)
    lines[2:5] = [f"{board}\n", f"{pot}\n", f"{effective_stacks}\n"]

    # Write the modified lines back to the file
    with open(SOLVER_INPUT_FILE, "w") as file:
        file.writelines(lines)

    target_exploitability = 0.025 * pot
    print(f"Solving! Targeting exploitability of {target_exploitability}")
    # Execute the script directly
    result = subprocess.run([SOLVER_SCRIPT], check=True)

    with open(SOLVER_OUTPUT_PATH, 'r') as file:
        output = json.load(file)
          
    return output

def clear_screen():
    if os.name == 'nt':  # For Windows
        os.system('cls')
    else:  # For Unix-like systems (Linux, macOS, etc.)
        os.system('clear')
