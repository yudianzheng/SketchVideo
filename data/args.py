import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Preprocess image sequence')
parser.add_argument(
    '--data_folder', type=Path, default=Path('../data/'), help='folder to process')
parser.add_argument(
    '--bg_folder', type=Path, default=Path('../data/'), help='folder to process')
parser.add_argument(
    '--num_frames', type=int, default=51, help='maximun num of frames')
parser.add_argument(
    '--evaluate_at', type=Path, default=Path('10000'), help='epoch evaluating at')
parser.add_argument(
    '--iters_num', type=int, default=10001, help='total epoch number')
parser.add_argument(
    '--result_folder', type=Path, default=Path('../data/'), help='folder to save')
parser.add_argument(
    '--resx', type=int, default=1080, help='folder to save')
parser.add_argument(
    '--resy', type=int, default=1080, help='folder to save')

args = parser.parse_args()

data_folder = args.data_folder
bg_folder = args.bg_folder
maximum_number_of_frames = args.num_frames
evaluate_every = args.evaluate_at
result_folder = args.result_folder
resx = args.resx
resy = args.resy
iters_num = args.iters_num

# Specify the file path
file_path = "config/config.json"

# Open the file in read mode
with open(file_path, "r") as f:
    # Read the lines of the file
    lines = f.readlines()

# Loop over the lines and replace any line that includes "data_folder"
new_lines = []
for line in lines:
    if "data_folder" in line:
        # Replace the line with the new sentence
        new_lines.append(f"    \"data_folder\": \"{data_folder}\",\n")
    elif "bg_folder" in line:
        new_lines.append(f"    \"bg_folder\": \"{bg_folder}\",\n")
    elif "maximum_number_of_frames" in line:
        new_lines.append(f"    \"maximum_number_of_frames\": {maximum_number_of_frames},\n")
    elif "evaluate_every" in line:
        new_lines.append(f"    \"evaluate_every\": \"{evaluate_every}\",\n")
    elif "results_folder_name" in line:
        new_lines.append(f"    \"results_folder_name\": \"{result_folder}\",\n")
    elif "resx" in line:
        new_lines.append(f"    \"resx\": {resx},\n")
    elif "resy" in line:
        new_lines.append(f"    \"resy\": {resy},\n")
    elif "iters_num" in line:
        new_lines.append(f"    \"iters_num\": {iters_num},\n")
    else:
        # Leave the line unchanged
        new_lines.append(line)

# Open the file in write mode and write the new lines
with open(file_path, "w") as f:
    f.writelines(new_lines)
