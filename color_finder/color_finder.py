import cv2
import numpy as np
from background_remover import (
    ImageProcessor,
)  # Import from your background removal script
from scipy.spatial import KDTree
from sklearn.cluster import KMeans


class ColorDictionary:
    """Encapsulates a color dictionary and provides KDTree-based lookup."""

    def __init__(self, color_dict):
        self.color_dict = color_dict
        self.kd_tree = self._build_kd_tree()

    def _build_kd_tree(self):
        color_values = list(self.color_dict.values())
        return KDTree(color_values)

    def find_closest_color(self, rgb_value):
        _, index = self.kd_tree.query(rgb_value)
        return list(self.color_dict.keys())[index]


class ColorFinder:
    """Integrates color finding with image processing and background removal."""

    def __init__(self, image_path, color_dict, removal_strategy):
        self.image_path = image_path
        self.color_dict = ColorDictionary(color_dict)
        self.removal_strategy = removal_strategy

    def process_image(self):
        # Use the ImageProcessor to read and convert the image to RGB
        image = ImageProcessor.read_image(self.image_path)
        image_rgb = ImageProcessor.convert_to_rgb(image)

        # Remove the background from the image
        image_no_bg = self.removal_strategy.remove_background(image_rgb)

        # Find the dominant color in the foreground
        dominant_color = self._find_dominant_color(image_no_bg)
        closest_color_name = self.color_dict.find_closest_color(dominant_color)

        return closest_color_name

    def _find_dominant_color(self, image):
        """Finds the most dominant color in the image using k-means clustering."""
        # Reshape the image to be a list of pixels
        pixels = image.reshape((-1, 3))

        # Remove black (background) pixels
        pixels = pixels[(pixels != [0, 0, 0]).all(axis=1)]

        # Use k-means clustering on the pixels to find the most dominant color
        if (
            len(pixels) > 0
        ):  # Ensure there are pixels left after removing the background
            cluster_count = 1
            clt = KMeans(n_clusters=cluster_count)
            clt.fit(pixels)

            # The cluster center with the highest count is the most dominant color
            dominant_color = clt.cluster_centers_[0].astype("uint8").tolist()
            return tuple(dominant_color)
        else:
            # Return a default color if no foreground pixels are found
            return (255, 255, 255)  # Example: white


# Example usage
if __name__ == "__main__":
    # Assume 'color_dict' is defined somewhere above or imported
    color_dict = {
        # Define your color dictionary here
        "AliceBlue": (240, 248, 255),
        # ... other colors
    }
    # Assume you have a way to create an instance of a removal strategy, for example:
    from background_remover import SimpleRemovalStrategy

    strategy = SimpleRemovalStrategy(threshold=240)

    finder = ColorFinder("path_to_your_image.jpg", color_dict, strategy)
    closest_color = finder.process_image()
    print(f"The closest color name is {closest_color}.")
