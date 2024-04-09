from setuptools import find_packages, setup

setup(
    name="color_finder",
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
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "color_finder=color_finder.color_finder:main",
        ],
    },
    author="Parth Sharma",
    author_email="reachout.parthsharma@gmail.com",
    description="A package for finding dominant colors in images and removing backgrounds.",
    keywords="image processing color detection background removal",
    # url="http://example.com/your_package_homepage",
    project_urls={
        "Source Code": "https://github.com/parth1993/pycolor_detection",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
