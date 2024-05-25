import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset_by_class(x, y, num_partitions):
    # Get unique class labels
    classes = np.unique(y)
    num_classes = len(classes)
    
    # Initialize partitions
    partitions = [[] for _ in range(num_partitions)]
    
    # Split data by class
    for class_label in classes:
        # Find indices of samples with the current class label
        indices = np.where(y == class_label)[0]
        
        # Split indices into equal partitions
        partition_indices = np.array_split(indices, num_partitions)
        
        # Assign each partition the corresponding samples
        for i, partition_idx in enumerate(partition_indices):
            partitions[i].extend(partition_idx)
    
    # Convert partition indices to data
    partition_data = []
    for partition in partitions:
        partition_x = x[partition]
        partition_y = y[partition]
        partition_data.append((partition_x, partition_y))
    
    return partition_data

# Example usage:
# partition_data = split_dataset_by_class(x_train, y_train, 3)
