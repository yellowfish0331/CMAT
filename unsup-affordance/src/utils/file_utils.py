import h5py
import matplotlib.pyplot as plt
import yaml

def store_or_update_dataset(h5_group, dataset_key, data, compression=None):
    """
    Store or update a dataset within an HDF5 group.
    
    If the dataset_key already exists, it is deleted first (to handle
    shape or type changes). Then a new dataset is created.
    
    If data is a list of Python strings, it is stored as a variable-length
    string array. Otherwise, it is stored as is.
    """
    import h5py

    # Remove existing dataset if present
    if dataset_key in h5_group:
        del h5_group[dataset_key]

    # If data is a list of strings, store as variable-length strings
    if (
        isinstance(data, list)
        and len(data) > 0
        and isinstance(data[0], str)
    ):
        dtype = h5py.special_dtype(vlen=str)
        dset = h5_group.create_dataset(
            dataset_key,
            shape=(len(data),),
            dtype=dtype,
            compression=compression
        )
        dset[:] = data
    else:
        # For numeric arrays, just store them directly
        h5_group.create_dataset(
            dataset_key,
            data=data,
            compression=compression
        )

def save_image(image, path):
    """Save image using matplotlib."""
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)