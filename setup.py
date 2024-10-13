from setuptools import setup, find_packages
import io
import os

#package details
NAME = "social_tools"
DESCRIPTION = "The goal is to provide a unified interface to interact with various social analysis tools"
URL = "https://github.com/instabaines/social_tools_lib/"
EMAIL = "amureridwan002@gmail.com"
AUTHOR = "Ridwan Amure"
VERSION ="0.1" 
REQUIRES_PYTHON = ">=3.10.0"
KEYWORDS = ["social tools", "sentiment analysis","toxicity","emotion analysis"]

cwd = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION
setup(
    name=NAME,
    version=VERSION,
    description = DESCRIPTION,
    author = AUTHOR,
    author_email=EMAIL,
    python_requires = REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests",]),
    license = "MIT",
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',   
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Natural Language :: English",
        "Operating System :: OS Independent"
    ],
    keywords= KEYWORDS
)