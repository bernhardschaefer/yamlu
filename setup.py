from setuptools import setup, find_packages

setup(
    name="yamlu",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "matplotlib",
        "numpy",
        "Pillow"
    ]
)
