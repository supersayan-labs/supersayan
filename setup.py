from setuptools import setup, find_packages

setup(
    name="supersayan",
    version="0.2.0",
    description="A Python frontend for Julia SupersayanTFHE Fully Homomorphic Encryption with PyTorch-style NN modules",
    author="Tom Massias Jurien de la Gravière, Franklin Tranié",
    author_email="tom.massiasjuriendelagraviere@epfl.ch, franklin.tranie@epfl.ch",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "julia",
    ],
    python_requires=">=3.9",
    classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
    ],
)
