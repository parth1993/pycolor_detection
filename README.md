# pycolor_detection v0.2.0
# Color Finder

The Color Finder package is a versatile Python library designed for image processing applications. It enables users to identify the dominant color within an image and perform background removal with both simple and advanced strategies. Whether you're developing applications related to image analysis, color detection, or just need a tool for processing images, Color Finder offers a comprehensive set of functionalities to meet your needs.

## Features

- **Dominant Color Detection**: Identify the most prevalent color in an image.
- **Background Removal**: Remove the background from an image using simple threshold-based techniques or advanced models like U2NET.
- **Color Dictionary**: Includes a predefined color dictionary mapping color names to RGB values, with support for easy customization.

## Installation

Ensure you have Python 3.6+ installed on your system. You can install the Color Finder package using pip:

```
pip install color-finder-package
```

## Usage

### Finding the Dominant Color

```python
from color_finder import ColorFinder, SimpleRemovalStrategy

# Path to your image
image_path = 'path/to/your/image.jpg'

# Initialize the ColorFinder with a simple background removal strategy
color_finder = ColorFinder(image_path, color_dict, SimpleRemovalStrategy(threshold=240))

# Process the image to find the closest named color
closest_color = color_finder.process_image()
print(f"The closest color name is {closest_color}.")
```

### Removing Image Background

```python
from color_finder import ImageProcessor, U2NETRemovalStrategy

# Load your U2NET model
model = load_u2net_model()
device = 'cuda'  # or 'cpu'

# Apply the U2NET strategy for background removal
strategy = U2NETRemovalStrategy(model=model, device=device)
image_no_bg = strategy.remove_background(ImageProcessor.read_image(image_path))

# Display the result
ImageProcessor.display_image(image_no_bg)
```

Note: Ensure to replace placeholders like load_u2net_model() with your actual model loading function.

## Contributing
Contributions to the Color Finder package are welcome! Whether it's adding new features, fixing bugs, or improving documentation, your help is appreciated.

To contribute:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.

Please ensure your code adheres to the project's coding standards and include tests for new features.

## License
This project is licensed under the MIT License - see the [**LICENSE**](https://github.com/parth1993/pycolor_detection/blob/main/LICENSE) file for details.


