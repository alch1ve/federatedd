import numpy as np

# Path to your .npz file
file_path = r"C:\Users\aldri\federatedd\dataset\CpE_Faculty_Members.npz"

# Load the dataset
data = np.load(file_path)

# Check the keys stored in the .npz file
keys = data.keys()
print("Keys in the .npz file:", keys)
