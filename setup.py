from setuptools import setup, find_packages

setup(
    name="aethelnet",
    version="0.1.0",
    description="A Continuous Liquid Graph Neural Network Framework",
    author="Aethelnet",
    author_email="nika.hrlyn@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchdiffeq>=0.2.3",
        "networkx>=3.0",
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "websockets>=11.0.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
