from setuptools import setup

setup(
    name="screenSim",
    version="0.1.0",
    author="Safaa Mouline",
    author_email="safaamouline@berkeley.edu",
    packages=["screenSim"],
    description="pooled genetic screen simulator for CRISPR perturbations",
    install_requires=[
        'numpy', 
        'pandas',
        "matplotlib",
    ]
)