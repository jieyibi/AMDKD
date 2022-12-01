import pickle
import numpy as np


def read_instance_pkl(instances_path):
    with open(instances_path, 'rb') as f:
        instances_data = pickle.load(f)

    return np.array(instances_data)

def read_instance_tsp(path):
    file = open(path, "r")
    lines = [ll.strip() for ll in file]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIMENSION"):
            dimension = int(line.split(':')[1])
        elif line.startswith('NODE_COORD_SECTION'):
            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=float)
            i = i + dimension

        i += 1

    original_locations = locations[:, 1:]
    original_locations = np.expand_dims(original_locations, axis=0)

    locations = original_locations / original_locations.max()  # Scale location coordinates to [0, 1]
    print(original_locations.max() )
    return original_locations, locations, original_locations.max()