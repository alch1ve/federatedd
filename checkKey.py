import numpy as np

# Load the .npz file
file_path = "C:/Users/aldri/federatedd/dataset/Class_1.npz"
data = np.load(file_path)

# List all the arrays stored in the .npz file
array_names = data.files
print("Keys of arrays in the .npz file:", array_names)

# Specify the keys for y_train and y_test
y_train_key = 'y_train'  # Update this key if it's different
y_test_key = 'y_test'    # Update this key if it's different

# Check if the specified keys exist in the .npz file
if y_train_key in array_names and y_test_key in array_names:
    # Access the arrays
    y_train = data[y_train_key]
    y_test = data[y_test_key]

    # Print the labels
    print("y_train labels:", y_train)
    print("y_test labels:", y_test)
else:
    print("The specified keys do not exist in the .npz file.")
