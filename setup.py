"""This file is to allow this repository to be published as a pip module, such
that people can install it with: `pip install networkx-to-lava-nc`.

You can ignore it.
"""
import setuptools

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="networkx-to-lava-nc-snn",
    version="0.0.1",
    author="a-t-0",
    author_email="author@example.com",
    description="Converts networkx graphs representing spiking neural networks"
    + " (SNN)s of LIF neruons, into runnable Lava SNNs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/a-t-0/networkx-to-lava-nc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: AGPL3",
        "Operating System :: OS Independent",
    ],
)
