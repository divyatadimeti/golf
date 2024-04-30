import os

def list_files(directory_path):
    """List all files in the given directory, sorted numerically."""
    file_list = []
    try:
        # Iterate through all entries in the directory
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.mp4'):  # Check for .mp4 files
                    file_list.append(entry.name)
    except FileNotFoundError:
        print(f"Error: The directory {directory_path} does not exist.")
    except PermissionError:
        print(f"Error: Permission denied to access the directory {directory_path}.")

    # Extracting numbers and sorting them
    numbers = sorted(int(file.split('.')[0]) for file in file_list)
    return numbers

def find_missing(numbers):
    """Find missing numbers in a sorted list."""
    if not numbers:
        return []
    start, end = numbers[0], numbers[-1]
    full_set = set(range(start, end + 1))
    missing = sorted(full_set - set(numbers))
    return missing

# Directory path
directory_path = '/home/dt2760/golf/data/videos_160_pose_estimation'

# Get the list of files
file_numbers = list_files(directory_path)

# Find missing numbers
missing_numbers = find_missing(file_numbers)

# Print the results
print("Sorted File Numbers:")
print(file_numbers)

print("\nMissing Numbers:")
print(missing_numbers)

