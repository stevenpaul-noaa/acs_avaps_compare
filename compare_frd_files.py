import numpy as np
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
import re
import datetime

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
                    # Check for a data line (parts[0] is index, ensure enough columns exist)
                    if len(parts) > 9 and parts[0].isdigit():
                        try:
                            # Time is at index 1 and is the key for combining files
                            time_s = float(parts[1])

                            for label, index in COLUMNS_TO_EXTRACT.items():
                                value = float(parts[index])
                                # Only store if the value is not the invalid marker
                                if abs(value - INVALID_DATA_VALUE) > 0.1:
                                    # Use a rounded time key for alignment since t(s) has varying precision
                                    # Rounding to 2 decimal places for 0.25 second precision data (1/4 second)
                                    data[round(time_s, 2)][label] = value
                        except (ValueError, IndexError):
                            # Skip lines that can't be parsed
                            continue
    except FileNotFoundError:
        # Changed to return an empty dict silently to avoid printing errors to console from within parse_frd_file
        return {}
    except Exception as e:
        # Changed to return an empty dict silently
        return {}
    return dict(data)

def compare_data(data1: Dict[float, Dict[str, float]],
                 data2: Dict[float, Dict[str, float]],
                 parameter: str,
                 threshold: float) -> Tuple[int, float, float, float, float, int, float, List[float]]:
    """
    Compares one parameter between two datasets (File 1 - File 2),
    calculates statistics, and checks against the threshold.
    Returns diffs list as well for global aggregation.
    """
    diffs = []

    # Identify all unique time steps present in both files for this comparison
    common_time_steps = sorted(set(data1.keys()) & set(data2.keys()))

    for time_s in common_time_steps:
        # Safely retrieve data points
        val1 = data1[time_s].get(parameter)
        val2 = data2[time_s].get(parameter)

        # Only compare if both values exist (were not -999.0)
        if val1 is not None and val2 is not None:
            # Calculation: File 1 (AVAPS) - File 2 (ACS)
            diff = val1 - val2
            diffs.append(diff)

    if not diffs:
        # Return nan for statistics if no comparable values found
        return 0, np.nan, np.nan, np.nan, np.nan, 0, 0.0, []

    diffs_array = np.array(diffs)

    # Calculate statistics
    total_values = len(diffs_array)
    mean_diff = np.mean(diffs_array)
    min_diff = np.min(diffs_array)
    max_diff = np.max(diffs_array)
    std_dev = np.std(diffs_array)

    # Count values within threshold (absolute difference <= threshold)
    within_threshold = np.sum(np.abs(diffs_array) <= threshold)
    percent_within = (within_threshold / total_values) * 100 if total_values > 0 else 0.0

    return total_values, mean_diff, min_diff, max_diff, std_dev, within_threshold, percent_within, diffs

def format_output(param: str, stats: Tuple[int, float, float, float, float, int, float],
                  unit: str) -> str:
    """Formats the statistical output string to match the requested format."""
    total, mean, min_d, max_d, std, within, percent = stats

    # Determine the label
    label_map = {
        'P': 'Pressure',
        'T': 'Temperature',
        'RH': 'Humidity',
        'U': 'U Winds',
        'V': 'V Winds'
    }
    label = label_map.get(param, param)

    # Handle the case of no comparable values
    if total == 0:
        return f"\nAVAPS - ACS {label} ({unit}):\n  No comparable data points found.\n"

    # Format the output strings exactly as requested
    output = f"\nAVAPS - ACS {label}:\n"
    output += f"  Total values            : {total}\n"
    output += f"  Mean difference         : {mean:.4f}\n"
    output += f"  Min/Max difference      : {min_d:.4f} / {max_d:.4f}\n"
    output += f"  Std dev                 : {std:.4f}\n"
    output += f"  Within threshold        : {within} ({percent:.2f}%)\n"

    return output

def extract_timestamp(filename: str) -> str:
    """Extracts the timestamp (YYYYMMDD_HHMMSS) from the filename."""
    # Regex for AVAPS file: DYYYYMMDD_HHMMSS_PQC.frd
    avaps_match = re.search(r'D(\d{8}_\d{6})_PQC\.frd', filename)
    if avaps_match:
        return avaps_match.group(1)

    # Regex for ACS file: HX_Melissa-YYYYMMDDH1-15-YYYYMMDDTHHMMSS-2QC.frd
    # Extracting the YYYYMMDDTHHMMSS part
    acs_match = re.search(r'-\d{8}T(\d{6})', filename)
    if acs_match:
        # For ACS, the date is in the first part of the filename
        date_match = re.search(r'-(\d{8})H1', filename)
        if date_match:
             return f"{date_match.group(1)}_{acs_match.group(1)}"


    return None

def main():
    """Parses arguments and runs the file comparison."""
    parser = argparse.ArgumentParser(
        description="Compares specified parameters between pairs of .frd files in a directory (File 1 - File 2)."
    )
    parser.add_argument(
        "directory_path",
        type=str,
        help="Path to the directory containing the .frd files."
    )
    args = parser.parse_args()

    directory_path = args.directory_path

    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"comparison_summary_{timestamp}.txt"

    with open(output_filename, 'w') as outfile:
        def print_to_file(text):
            outfile.write(text + '\n')

        avaps_files = {}
        acs_files = {}

        # Collect files and group by timestamp
        for filename in os.listdir(directory_path):
            if filename.endswith(".frd"):
                file_path = os.path.join(directory_path, filename)
                timestamp = extract_timestamp(filename)
                if timestamp:
                    if filename.startswith('D'): # Assuming AVAPS files start with 'D'
                        avaps_files[timestamp] = file_path
                    else: # Assuming ACS files do not start with 'D'
                        acs_files[timestamp] = file_path

        # Find matching pairs based on timestamp, allowing for +/- 1 second difference
        # Create a mapping from ACS timestamp to potential AVAPS timestamps
        acs_timestamp_mapping = {}
        for acs_ts in acs_files.keys():
            try:
                # Convert timestamp to integer seconds for comparison
                date_part, time_part = acs_ts.split('_')
                acs_time_in_seconds = int(time_part)
                acs_timestamp_mapping[acs_ts] = []
                # Look for AVAPS timestamps within +/- 1 second
                for avaps_ts in avaps_files.keys():
                    avaps_date_part, avaps_time_part = avaps_ts.split('_')
                    if avaps_date_part == date_part:
                        avaps_time_in_seconds = int(avaps_time_part)
                        if abs(avaps_time_in_seconds - acs_time_in_seconds) <= 1:
                            acs_timestamp_mapping[acs_ts].append(avaps_ts)
            except (ValueError, IndexError):
                print_to_file(f"Warning: Could not parse timestamp from ACS file: {acs_files[acs_ts]}")
                continue

        compared_pairs = set() # To avoid duplicate comparisons
        global_diffs: Dict[str, List[float]] = defaultdict(list)
        file_count = 0

        # Iterate through the ACS timestamps and their potential AVAPS matches
        for acs_ts, avaps_ts_list in acs_timestamp_mapping.items():
            if acs_ts in acs_files: # Ensure the ACS file still exists in our dictionary
                if avaps_ts_list:
                    # Prioritize an exact match if available
                    exact_match_ts = None
                    for avaps_ts in avaps_ts_list:
                        if avaps_ts == acs_ts:
                            exact_match_ts = avaps_ts
                            break

                    if exact_match_ts:
                        avaps_ts_to_compare = exact_match_ts
                    else:
                        # If no exact match, pick the first one in the list (could be off by 1 second)
                        avaps_ts_to_compare = avaps_ts_list[0]


                    if avaps_ts_to_compare in avaps_files:
                         file1_path = avaps_files[avaps_ts_to_compare]
                         file2_path = acs_files[acs_ts]

                         # Create a unique key for the pair regardless of order
                         pair_key = tuple(sorted((file1_path, file2_path)))

                         if pair_key not in compared_pairs:
                            print_to_file(f"\nComparing pair based on timestamp {acs_ts}:")
                            print_to_file(f"  File 1 (AVAPS): {os.path.basename(file1_path)}")
                            print_to_file(f"  File 2 (ACS): {os.path.basename(file2_path)}")

                            data1 = parse_frd_file(file1_path)
                            data2 = parse_frd_file(file2_path)

                            results = []

                            # Check if files were successfully loaded
                            if not data1 or not data2:
                                print_to_file("Comparison aborted due to file loading errors.")
                            else:
                                 file_count += 1
                                 # Map parameters to units for display
                                 unit_map = {
                                     'P': 'mb',
                                     'T': 'C',
                                     'RH': '%',
                                     'U': 'm/s',
                                     'V': 'm/s'
                                 }

                                 # Compare each parameter and accumulate global diffs
                                 for param, threshold in THRESHOLDS.items():
                                     stats = compare_data(data1, data2, param, threshold)
                                     unit = unit_map.get(param, '')
                                     results.append(format_output(param, stats[:7], unit)) # Pass only first 7 stats for formatting
                                     global_diffs[param].extend(stats[7]) # Extend global diffs with the list of diffs

                                 # Print results for the current pair to file
                                 print_to_file(f"\n{'='*20} Comparison Results {'='*20}")
                                 print_to_file(f"Comparing: {os.path.basename(file1_path)} - {os.path.basename(file2_path)}")
                                 print_to_file('-'*58)
                                 for result in results:
                                     print_to_file(result)
                                 print_to_file('='*58)

                            compared_pairs.add(pair_key) # Mark this pair as compared
                    else:
                         print_to_file(f"Warning: Could not find a matching AVAPS file for ACS timestamp {acs_ts}")
                else:
                    print_to_file(f"Warning: No potential AVAPS matches found for ACS file: {os.path.basename(acs_files[acs_ts])} with timestamp {acs_ts}")


        if not compared_pairs:
            print_to_file("No matching file pairs found in the directory.")
        else:
            # Calculate and print global summary to file
            print_to_file(f"\n\n=== GLOBAL SUMMARY ACROSS {file_count} FILE(S) ===")

            unit_map = {
                'P': 'mb',
                'T': 'C',
                'RH': '%',
                'U': 'm/s',
                'V': 'm/s'
            }

            for param, threshold in THRESHOLDS.items():
                diffs = global_diffs[param]
                if not diffs:
                    print_to_file(f"\nAVAPS - ACS {param}:")
                    print_to_file("  No comparable data points found globally.")
                    continue

                global_diffs_array = np.array(diffs)

                total_values = len(global_diffs_array)
                mean_diff = np.mean(global_diffs_array)
                min_diff = np.min(global_diffs_array)
                max_diff = np.max(global_diffs_array)
                std_dev = np.std(global_diffs_array)
                within_threshold = np.sum(np.abs(global_diffs_array) <= threshold)
                percent_within = (within_threshold / total_values) * 100 if total_values > 0 else 0.0

                label_map = {
                    'P': 'Pressure',
                    'T': 'Temperature',
                    'RH': 'Humidity',
                    'U': 'U Winds',
                    'V': 'V Winds'
                }
                label = label_map.get(param, param)
                unit = unit_map.get(param, '')


                print_to_file(f"\nAVAPS - ACS {label}:")
                print_to_file(f"  Total values        : {total_values}")
                print_to_file(f"  Mean difference     : {mean_diff:.4f}")
                print_to_file(f"  Min/Max difference  : {min_diff:.4f} / {max_diff:.4f}")
                print_to_file(f"  Std dev             : {std_dev:.4f}")
                print_to_file(f"  Within threshold    : {within_threshold} ({percent_within:.2f}%)")

        print(f"Comparison summary saved to {output_filename}")


if __name__ == "__main__":
    main()
