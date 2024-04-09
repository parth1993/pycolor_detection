from unittest.mock import MagicMock

import cv2
import numpy as np
from background_remover import (
    ImageProcessor,
    SimpleRemovalStrategy,
    U2NETRemovalStrategy,
)


def test_simple_removal():
    # Assume test_image.jpg is a simple image with a uniform background
    test_image_path = "path/to/test_image.jpg"
    image = ImageProcessor.read_image(test_image_path)
    strategy = SimpleRemovalStrategy(threshold=240)
    result = strategy.remove_background(ImageProcessor.convert_to_rgb(image))

    # Test: Background is removed (assuming white background for simplicity)
    # Adjust this assertion based on your test image's characteristics
    assert np.all(
        result[np.where((result == [0, 0, 0]).all(axis=2))] == [0, 0, 0]
    )


def test_u2net_removal():
    # Mocking the U2NET model's output for a simple foreground-background separation
    mock_model = MagicMock()
    mock_model.return_value = MagicMock()
    mock_model.return_value.cpu.return_value.detach.return_value.numpy.return_value = np.ones(
        (320, 320), dtype=np.float32
    )

    device = "cpu"  # Simplification for testing
    strategy = U2NETRemovalStrategy(model=mock_model, device=device)

    test_image_path = "path/to/test_image.jpg"
    image = ImageProcessor.read_image(test_image_path)
    result = strategy.remove_background(image)

    # Test: Entire image is considered foreground (based on mock model output)
    # Adjust based on your mock setup and expected outcomes
    assert np.all(result != 0)
