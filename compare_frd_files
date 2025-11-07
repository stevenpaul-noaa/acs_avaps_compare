import numpy as np
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# Define the file paths and thresholds
# NOTE: These paths must exactly match the names of your .frd files
FILE1_PATH = 'D20250123_203710_PQC.frd'
FILE2_PATH = 'AR2025-20250123N1-06-20250123T203710-5QC.frd'

# Comparison thresholds as requested: Pressure 1.0, Temperature 0.2, Humidity 5.0, U Winds 1.0, V Winds 1.0
THRESHOLDS = {
    'P': 1.0,
    'T': 0.2,
    'RH': 5.0,
    'U': 1.0,
    'V': 1.0
}

# The columns of interest and their 0-based index in the data section of the .frd file
# Data columns: IX, t (s), P (mb), T (C), RH (%), Z (m), WD, WS (m/s), U (m/s), V (m/s), ...
COLUMNS_TO_EXTRACT = {
    'P': 2,
    'T': 3,
    'RH': 4,
    'U': 8,
    'V': 9,
}

# Value used to indicate invalid data in the .frd files
INVALID_DATA_VALUE = -999.0

def parse_frd_file(file_path: str) -> Dict[float, Dict[str, float]]:
    """Parses a .frd file, extracting specified data keyed by time (t(s))."""
    data = defaultdict(dict)
    try:
        with open(file_path, 'r') as f:
            data_section_started = False
            for line in f:
                line = line.strip()
                
                # Check for the start of the data section (line beginning with column names)
                if line.startswith('IX'):
                    data_section_started = True
                    continue
                
                if data_section_started:
                    parts = line.split()
                    if len(parts) > 9 and parts[0].isdigit(): # Check for a valid data line
                        try:
                            # Time is always at index 1 and is the key for combining files
                            time_s = float(parts[1])
                            
                            for label, index in COLUMNS_TO_EXTRACT.items():
                                value = float(parts[index])
                                # Only store if the value is not the invalid marker
                                if abs(value - INVALID_DATA_VALUE) > 0.1:
                                    # Use a rounded time key for alignment since t(s) has varying precision
                                    data[round(time_s, 2)][label] = value
                        except (ValueError, IndexError):
                            # Skip lines that can't be parsed (e.g., header lines in data section or malformed lines)
                            continue
    except FileNotFoundError:
        print(f"Error: Input file not found: {file_path}")
        return {}
    return dict(data)

def compare_data(data1: Dict[float, Dict[str, float]], 
                 data2: Dict[float, Dict[str, float]], 
                 parameter: str, 
                 threshold: float) -> Tuple[int, float, float, float, float, int, float]:
    """
    Compares one parameter between two datasets, calculates statistics,
    and checks against the threshold.
    """
    diffs = []
    
    # Iterate over common time keys
    for time_s in data1.keys():
        if time_s in data2:
            val1 = data1[time_s].get(parameter)
            val2 = data2[time_s].get(parameter)
            
            # Only compare if both values exist (were not -999.0) for this parameter
            if val1 is not None and val2 is not None:
                # We want AVAPS - ACS, which corresponds to File 1 - File 2
                diff = val1 - val2
                diffs.append(diff)

    if not diffs:
        # Return nan for statistics if no comparable values found
        return 0, np.nan, np.nan, np.nan, np.nan, 0, 0.0

    diffs_array = np.array(diffs)
    
    # Calculate statistics
    total_values = len(diffs_array)
    mean_diff = np.mean(diffs_array)
    min_diff = np.min(diffs_array)
    max_diff = np.max(diffs_array)
    std_dev = np.std(diffs_array)
    
    # Count values within threshold
    within_threshold = np.sum(np.abs(diffs_array) <= threshold)
    percent_within = (within_threshold / total_values) * 100 if total_values > 0 else 0.0

    return total_values, mean_diff, min_diff, max_diff, std_dev, within_threshold, percent_within

def format_output(param: str, stats: Tuple[int, float, float, float, float, int, float], 
                  unit: str, threshold: float) -> str:
    """Formats the statistical output string."""
    total, mean, min_d, max_d, std, within, percent = stats
    
    # Determine the label
    if param == 'P':
        label = 'Pressure'
    elif param == 'T':
        label = 'Temperature'
    elif param == 'RH':
        label = 'Humidity'
    elif param == 'U':
        label = 'U Winds'
    elif param == 'V':
        label = 'V Winds'
    else:
        label = param

    # Handle the case of no comparable values
    if total == 0:
        return f"\nAVAPS - ACS {label} ({unit}):\n  No comparable data points found.\n"
        
    output = f"\nAVAPS - ACS {label} ({unit}):\n"
    output += f"  Total values            : {total}\n"
    output += f"  Mean difference         : {mean:.4f}\n"
    output += f"  Min/Max difference      : {min_d:.4f} / {max_d:.4f}\n"
    output += f"  Std dev                 : {std:.4f}\n"
    output += f"  Threshold (Absolute)    : {threshold:.2f}\n"
    output += f"  Within threshold        : {within} ({percent:.2f}%)\n"
    
    return output

def main_compare(file1_path: str, file2_path: str, thresholds: Dict[str, float]):
    """Main function to perform comparison and print results."""
    
    print(f"Parsing File 1 (AVAPS-like): {os.path.basename(file1_path)}")
    data1 = parse_frd_file(file1_path)
    print(f"Parsing File 2 (ACS-like): {os.path.basename(file2_path)}")
    data2 = parse_frd_file(file2_path)
    
    results = []
    
    # Check if files were successfully loaded
    if not data1 or not data2:
        print("\nComparison aborted due to file loading errors.")
        return

    # Compare each parameter
    for param, threshold in thresholds.items():
        stats = compare_data(data1, data2, param, threshold)
        
        # Determine the appropriate unit for display
        if param == 'P':
            unit = 'mb'
        elif param == 'T':
            unit = 'C'
        elif param == 'RH':
            unit = '%'
        elif param in ['U', 'V']:
            unit = 'm/s'
        else:
            unit = ''

        results.append(format_output(param, stats, unit, threshold))

    # Print results
    print(f"\n{'='*20} Comparison Results {'='*20}")
    print(f"File 1 (AVAPS-like): {os.path.basename(file1_path)}")
    print(f"File 2 (ACS-like): {os.path.basename(file2_path)}")
    print('-'*58)
    for result in results:
        print(result)
    print('='*58)

if __name__ == "__main__":
    main_compare(FILE1_PATH, FILE2_PATH, THRESHOLDS)
