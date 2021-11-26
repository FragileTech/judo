"""judo package installation metadata."""
from importlib.machinery import SourceFileLoader
from pathlib import Path

from setuptools import find_packages, setup


version = SourceFileLoader(
    "judo.version",
    str(Path(__file__).parent / "judo" / "version.py"),
).load_module()

with open(Path(__file__).with_name("README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="judo",
    description="API and data structures for efficient AI research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    version=version.__version__,
    license="MIT",
    author="FragileTech",
    author_email="info@fragile.tech",
    url="https://github.com/FragileTech/judo",
    keywords=["Machine learning", "artificial intelligence"],
    test_suite="tests",
    tests_require=["pytest>=5.3.5", "hypothesis>=5.6.0"],
    extras_require={},
    install_requires=[],
    package_data={"": ["README.md"], "judo": ["config.yml"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
    ],
)
