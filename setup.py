from setuptools import setup

setup(
    name="screenSim",
    version="0.0.5",
    description="simulator for count data of CRISPR screens",
    url = "https://github.com/smouline/Screen-Simulator",
    author="Safaa Mouline",
    author_email="safaamouline@berkeley.edu",
    packages=["screenSim"],
    install_requires=[
        'numpy', 
        'pandas',
        "matplotlib",
    ]
)