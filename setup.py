import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="idepthdataset",
    py_modules=["idd"],
    version="0.1",
    description="Iphone depth dataset",
    readme="README.md",
    python_requires=">=3.7",
    author="Anh Vo Trann Hai",
    url="https://github.com/anhvth/IDepthDataset",
    license="MIT",
    # packages=find_packages(exclude=["tests*"]),
    install_requires=[
        # str(r)
        # for r in pkg_resources.parse_requirements(
        #     open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        # )
    ],
    include_package_data=True,
    extras_require={"dev": ["pytest"]},
)
