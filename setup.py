from setuptools import setup, find_packages

setup(
    name="pore_insight",  
    version="0.1.0",  
    description="A Python package for pore size distribution analysis",
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown",
    author="Gergo Ignacz",
    author_email="gergo.ignacz@kaust.edu.sa",
    url="https://github.com/ignaczgerg/pore_insight",  
    license="MIT",  
    packages=find_packages(),  
    install_requires=[
        "numpy",
        "scipy",
        "rdkit"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)