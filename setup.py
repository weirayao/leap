import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README.mf file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
  long_description = f.read()

setup(
    name="leap", # Replace with your own username
    version="0.0.1",
    author="Weiran Yao, Yuewen Sun, Alex Ho, Kun Zhang",
    author_email="leap@googlegroups.com",
    description="The leap is a tool for discovering latent temporal causal factors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/weirayao/leap",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires = [
        "pytorch-lightning==1.2.7",
        "torch==1.8.1",
        "disentanglement-lib==1.4",
        "torchvision",
        "torchaudio",
        "h5py",
        "ipdb",
        "opencv-python",
        "pymunk"
    ],
    tests_require=[
        "pytest"
    ],
)