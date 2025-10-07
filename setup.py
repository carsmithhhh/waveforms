from setuptools import setup, find_packages

setup(
    name="waveforms",
    version="0.1.0",
    author="Carolyn Smith",
    author_email="carsmith@stanford.edu",
    description="Stochastic LArTPC optical waveform simulator.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/carsmithhhh/waveforms",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch"
    ],
    python_requires=">=3.8",
)