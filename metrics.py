import numpy as np

# Initialize a list to hold all the batches
batches = []

# Open and read from the file
with open('evaluation_results_pose_estimation_adj3.txt', 'r') as file:
    lines = file.readlines()
    # Parse each line
    for line in lines:
        # Find the part of the string that contains the numbers (brackets part)
        start = line.find('[') + 1
        end = line.find(']')
        if start == 0 or end == -1:
            continue  # This skips lines without brackets
        number_str = line[start:end]
        try:
            # Create a list of integers from this substring
            numbers = list(map(int, number_str.split()))
            batches.append(numbers)
        except ValueError:
            continue  # This skips lines that cannot be converted to integer lists

# Convert list of batches into a NumPy array for easier manipulation
if batches:
    batches = np.array(batches)
else:
    raise ValueError("No valid data found in the file.")

# Calculate total number of batches
total_batches = batches.shape[0]

# Calculate accuracy for each class
class_accuracies = batches.sum(axis=0) / total_batches

# Calculate overall accuracy
overall_accuracy = batches.sum() / batches.size

# Create confusion matrix (Here assuming binary classification per class: correct (1) or incorrect (0))
confusion_matrices = np.zeros((8, 2, 2), dtype=int)  # There are 8 classes, each with a 2x2 confusion matrix

for i in range(8):
    confusion_matrices[i, 0, 0] = np.sum(batches[:, i] == 0)  # True Negatives: correctly identified as 0
    confusion_matrices[i, 0, 1] = np.sum(batches[:, i] == 1)  # False Positives: incorrectly identified as 1
    confusion_matrices[i, 1, 0] = np.sum(batches[:, i] == 1)  # False Negatives: incorrectly identified as 0
    confusion_matrices[i, 1, 1] = np.sum(batches[:, i] == 0)  # True Positives: correctly identified as 1

# Print class accuracies
class_labels = ["Address (A)", "Toe-up (TU)", "Mid-backswing (MB)", "Top (T)", "Mid-downswing (MD)", "Impact (I)", "Mid-follow-through (MFT)", "Finish (F)"]
for label, accuracy in zip(class_labels, class_accuracies):
    print(f"Accuracy for {label}: {accuracy:.2%}")

# Print overall accuracy
print(f"Overall Accuracy: {overall_accuracy:.2%}")

# Output confusion matrices for each class
for i, label in enumerate(class_labels):
    print(f"Confusion Matrix for {label}:")
    print(confusion_matrices[i])
    print()

