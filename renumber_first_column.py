#!/usr/bin/env python3
"""
Script to renumber the first column of a telemetry data file starting from 0.
Preserves exact spacing and formatting of all other columns.
"""

import re
import sys

def renumber_first_column(file_path):
    """
    Renumber the first column of the data file starting from 0.
    Preserves exact spacing and formatting.
    """
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Process each line
    new_lines = []
    counter = 0
    
    for line in lines:
        # Keep comment lines and empty lines unchanged
        if line.strip().startswith('#') or line.strip() == '':
            new_lines.append(line)
            continue
        
        # For data lines, replace the first number
        # Pattern: capture leading spaces, the first number, separator spaces, and rest of line
        match = re.match(r'^(\s*)(\d+)(\s+)(.*)', line)
        if match:
            leading_spaces = match.group(1)
            first_number = match.group(2)
            separator_spaces = match.group(3)
            rest_of_line = match.group(4)  # This includes the newline
            
            # Calculate the width needed for the original number
            original_width = len(first_number)
            
            # Create new line with counter, preserving the original number width
            new_number = str(counter).rjust(original_width)
            new_line = f"{leading_spaces}{new_number}{separator_spaces}{rest_of_line}"
            new_lines.append(new_line)
            counter += 1
        else:
            # If line doesn't match expected pattern, keep it unchanged
            new_lines.append(line)
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Successfully renumbered {counter} data lines.")
    print(f"First column now starts from 0 and goes to {counter-1}.")

if __name__ == "__main__":
    file_path = "/home/first/yjkim/PLred/PLred/tutorials/data/fastcam/cropped_palila_15:05:10.002607418.txt"
    
    # Create backup first
    import shutil
    backup_path = file_path + ".backup"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Renumber the file
    renumber_first_column(file_path)
    
    # Show first few lines to verify
    print("\nFirst few data lines after renumbering:")
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_count = 0
        for i, line in enumerate(lines):
            if not line.strip().startswith('#') and line.strip():
                print(f"Line {i+1}: {line.rstrip()}")
                data_count += 1
                if data_count >= 5:  # Show first 5 data lines
                    break
