import numpy as np
from sklearn.model_selection import train_test_split

# Load datasets
npz_path_1 = r"C:\Users\aldri\federatedd\dataset\Class_1.npz"
npz_path_2 = r"C:\Users\aldri\federatedd\dataset\Class_2.npz"

data_1 = np.load(npz_path_1)
data_2 = np.load(npz_path_2)

# Extract data and labels from both datasets
x_test_1, y_test_1 = data_1["arr_0"], data_1["arr_1"]
x_test_2, y_test_2 = data_2["arr_0"], data_2["arr_1"]

# Combine data and labels
x_test_combined = np.concatenate((x_test_1, x_test_2), axis=0)
y_test_combined = np.concatenate((y_test_1, y_test_2), axis=0)

# Ensure labels are in the correct format
# Assuming labels are already in the correct format

# Split into test and validation sets while retaining the correspondence between data and labels
x_test_combined, _, y_test_combined, _ = train_test_split(x_test_combined, y_test_combined, test_size=0.2, random_state=42)

# Save the combined test set
np.savez("test_set.npz", x_test=x_test_combined, y_test=y_test_combined)
