from setuptools import setup, find_packages


setup(
    name="distributed-pytorch-template", # rename to your project name. This is used for pip install
    version="0.0.1",
    author="Chuyang Zhao",
    description="Distributed PyTorch Project Template",
    packages=find_packages(exclude=("tests*")),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "pandas",
        "tensorboard",
        "tqdm",
        "termcolor",
    ]
)