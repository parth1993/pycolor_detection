from setuptools import setup, find_packages

setup(
    name="color_finder_package",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "opencv-python-headless",
        "Pillow",
        "torch",
        "torchvision",
        "scikit-learn",
        "pytest",
        "pytest-mock",
    ],
    python_requires='>=3.6',
    entry_points={
        "console_scripts": [
            "color_finder=color_finder.color_finder:main",  # Example CLI entry point
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for finding dominant colors in images and removing backgrounds.",
    keywords="image processing color detection background removal",
    url="http://example.com/your_package_homepage",
    project_urls={
        "Source Code": "https://github.com/parth1993/color_finder_package",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
