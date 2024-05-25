import numpy as np

# Load the .npz file
data = np.load("C:/Users/aldri/federatedd/partitions/partition_1.npz")

# List all the arrays stored in the .npz file
array_names = data.files
print("Keys of arrays in the .npz file:", array_names)

# Specify the key of the array whose shape you want to know
desired_key = input("Enter the key of the array whose shape you want to know: ")

# Check if the specified key exists in the .npz file
if desired_key in array_names:
    array_shape = data[desired_key].shape
    print("Shape of the array {}: {}".format(desired_key, array_shape))
else:
    print("The specified key does not exist in the .npz file.")
