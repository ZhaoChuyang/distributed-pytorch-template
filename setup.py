import os
from setuptools import setup, find_packages


def get_project_info():
    """
    Get the project name and version from __init__.py
    You should not have more than one folders in the root directory with __init__.py
    """
    for folder in os.listdir("."):
        if folder.startswith("test"):
            continue
        if os.path.isdir(folder):
            files = os.listdir(folder)
            if "__init__.py" in files:
                proj_name = folder
                break
    
    init_py_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), proj_name, "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("VERSION")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return proj_name, version

proj_name, version = get_project_info()
name = "distributed-pytorch-template" if proj_name == "src" else proj_name
print(proj_name, version, name)
setup(
    name=name,
    version=version,
    author="Chuyang Zhao",
    description="Distributed PyTorch Project Template",
    packages=find_packages(exclude=("test*")),
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