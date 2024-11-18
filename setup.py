from setuptools import setup, find_packages

setup(
    name="aqmp-compressor",
    version="0.1.6",
    description="A Python package for compressing and decompressing images using OMP and wavelet transforms.",
    author="Emmanuel A. Tassone",
    author_email="emmanueltassone@gmail.com",
    url="https://github.com/Emmatassone/AQMP",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "Pillow",
        "PyWavelets",
        "anytree"
    ],
    entry_points={
        "console_scripts": [
            "aqmp=aqmp.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='<=3.11, >=3.9',
)