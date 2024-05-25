import os
import numpy as np
from sklearn.model_selection import train_test_split
import dataset

def split_dataset_by_class(dataset_x, dataset_y, num_partitions):
    # Get unique class labels and their counts
    unique_classes, class_counts = np.unique(dataset_y, return_counts=True)

    # Calculate number of classes per partition
    classes_per_partition = len(unique_classes) // num_partitions

    # Create empty lists to store partitioned data
    partitioned_data = []

    # Split dataset based on the partitioned class labels
    start_index = 0
    for i in range(num_partitions):
        # Calculate the end index for the current partition
        end_index = start_index + classes_per_partition

        # Check if this is the last partition
        if i == num_partitions - 1:
            end_index += len(unique_classes) % num_partitions

        # Get the classes for the current partition
        partition_classes = unique_classes[start_index:end_index]

        # Create a mask to filter the dataset based on the partitioned class labels
        mask = np.isin(dataset_y, partition_classes)

        # Apply the mask to partition the dataset
        x_partition = dataset_x[mask]
        y_partition = dataset_y[mask]

        # Add partitioned data to the list
        partitioned_data.append((x_partition, y_partition))

        # Update the start index for the next partition
        start_index = end_index

    return partitioned_data

# Define the path to your .npz dataset
npz_path = r"C:\Users\aldri\federatedd\dataset\CpE_Faculty_Members.npz"

# Load dataset
x_train, x_test, y_train, y_test = dataset.load_dataset_evaluate(npz_path, test_size=0.2)

# Encode labels as integers if needed
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Split the dataset into three partitions by class labels
partition_data = split_dataset_by_class(x_train, y_train_encoded, 2)

# Save each partition to a separate file
for i, (x_partition, y_partition) in enumerate(partition_data):
    partition_filename = f"partition_{i+1}.npz"
    np.savez(os.path.join("partitions", partition_filename), x=x_partition, y=y_partition)
    print(f"Partition {i+1} saved to {partition_filename}")
