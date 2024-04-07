import numpy as np
import pytest
from color_finder import ColorFinder, SimpleRemovalStrategy, ColorDictionary  # Adjust import paths as necessary
from background_removal import ImageProcessor  # Adjust import paths as necessary

@pytest.fixture
def color_dict():
    return {
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
        "Red": (255, 0, 0),
    }

@pytest.fixture
def simple_strategy():
    return SimpleRemovalStrategy(threshold=240)

@pytest.fixture
def sample_image_path():
    # Path to a sample image file for testing
    return "path/to/sample/image.jpg"

def test_read_image(sample_image_path):
    image = ImageProcessor.read_image(sample_image_path)
    assert image is not None

def test_convert_to_rgb(sample_image_path):
    image = ImageProcessor.read_image(sample_image_path)
    rgb_image = ImageProcessor.convert_to_rgb(image)
    # Assert that the conversion does not throw errors and returns an array
    assert isinstance(rgb_image, np.ndarray)

def test_simple_removal_strategy(sample_image_path, simple_strategy):
    image = ImageProcessor.read_image(sample_image_path)
    rgb_image = ImageProcessor.convert_to_rgb(image)
    no_bg_image = simple_strategy.remove_background(rgb_image)
    # Perform a basic check, for detailed checks consider mocking or using a known image
    assert no_bg_image is not None
    assert isinstance(no_bg_image, np.ndarray)

def test_color_dictionary_find_closest_color(color_dict):
    color_dictionary = ColorDictionary(color_dict)
    closest_color = color_dictionary.find_closest_color((254, 1, 1))  # Almost Red
    assert closest_color == "Red"

def test_color_finder_process_image(mocker, sample_image_path, color_dict, simple_strategy):
    # Mocking ImageProcessor to return a solid color image
    mocker.patch('background_removal.ImageProcessor.read_image', return_value=np.zeros((100, 100, 3), dtype=np.uint8))
    mocker.patch('background_removal.ImageProcessor.convert_to_rgb', side_effect=lambda x: x)
    
    # Assuming a simple image with a known dominant color
    color_finder = ColorFinder(sample_image_path, color_dict, simple_strategy)
    closest_color = color_finder.process_image()
    
    # Since the image is mocked to black, the closest color should be "Black"
    assert closest_color == "Black"
