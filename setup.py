from setuptools import setup

setup(
    name="screenSim",
    version="0.0.3",
    author="Safaa Mouline",
    author_email="safaamouline@berkeley.edu",
    packages=["screenSim"],
    description="simulator for CRISPRi screens",
    install_requires=[
        'numpy', 
        'pandas',
        "matplotlib",
    ]
)