from setuptools import setup, find_packages

setup(
    name="yamlu",
    version="0.0.4",
    description="yet another machine learning utility library",
    url="https://github.com/bernhardschaefer/yamlu",
    author="Bernhard Schäfer",
    license="Apache 2.0",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "matplotlib",
        "numpy",
        "Pillow"
    ]
)
