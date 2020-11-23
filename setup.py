from importlib.machinery import SourceFileLoader
from pathlib import Path

from setuptools import find_packages, setup


version = SourceFileLoader(
    "judo.version", str(Path(__file__).parent / "judo" / "version.py"),
).load_module()

with open(Path(__file__).with_name("README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="judo",
    description="Common API for working with tensors in the context of machine learning research ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    version=version.__version__,
    license="MIT",
    author="Guillem Duran Ballester",
    author_email="",
    url="https://github.com/FragileTech/judo",
    download_url="https://github.com/FragileTech/judo",
    keywords=["Machine learning", "artificial intelligence"],
    tests_require=["pytest>=5.3.5", "hypothesis>=5.6.0"],
    extras_require={},
    install_requires=[
        "numpy>=1.0.0",
        "pyyaml>=5.0.0",
        "xxhash>=1.1.0",
        "pillow-simd>=7.0.0",
        "networkx > 2.0.0",
    ],
    package_data={"": ["README.md"], "judo": ["judo/config.yml"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
    ],
)
