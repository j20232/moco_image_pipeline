import sys
from setuptools import setup, find_packages

sys.path.append("./mcp")
sys.path.append("./test")

with open("README.md") as f:
    long_description = f.read()

setup(
    name="mcp",
    version="0.1",
    author="mocobt",
    author_email="mocobt@gmail.com",
    description="image pipeline for kaggle",
    long_description=long_description,
    license="BSD 3-Clause",
    packages=find_packages(),
    test_suite="context.suite"
)
