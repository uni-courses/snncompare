"""This file is to allow this repository to be published as a pip module, such
that people can install it with: `pip install networkx-to-lava-nc`.

You can ignore it.
"""
import setuptools

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="snn_algo_compare",
    version="0.0.1",
    author="a-t-0",
    author_email="author@example.com",
    description="Runs an SNN algorithm and compares its results to its Neumann"
    + "implementation. Also supports adding adaptation and radiation to the "
    + "SNN.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/a-t-0/networkx-to-lava-nc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Natural Language :: English",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
