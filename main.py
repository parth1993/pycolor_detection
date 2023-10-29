import numpy as np
from scipy.spatial import KDTree

COLORS = {
    "white": [255, 255, 255],
    "black": [0, 0, 0],
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    # Add other colors if needed
}

# Build a KD-Tree
color_names = list(COLORS.keys())
color_values = np.array(list(COLORS.values()))
tree = KDTree(color_values)

def closest_color(rgb):
    """Find the closest color name for the given RGB value using KD-Tree."""
    dist, index = tree.query(rgb)
    return color_names[index]
