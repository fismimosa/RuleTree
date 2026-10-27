from setuptools import setup, find_packages
from codecs import open
from os import path

__author__ = "Cristiano Landi"
__license__ = "BSD-2-Clause"
__email__ = "cristiano.landi@phd.unipi.it"


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()


setup(
    name="RuleTree",
    python_requires='>=3.10.0',
    version="0.0.4-2",
    license="BSD-Clause-2",
    description="Interpretable decision trees built node by node with stumps of different kinds, for tabular data, time series and images",
    url="https://github.com/fismimosa/RuleTree",
    project_urls={
        "Documentation": "https://fismimosa.github.io/RuleTree/",
        "Source": "https://github.com/fismimosa/RuleTree",
        "Issues": "https://github.com/fismimosa/RuleTree/issues",
    },
    author="Cristiano Landi",
    author_email="cristiano.landi@phd.unipi.it",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    keywords="decision trees, interpretable machine learning, explainable AI, rules, time series, clustering, scikit-learn",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    extras_require={
        "shapelets": ["numba"],  # shapelet, proximity and partial pivot stumps
        "plot": ["matplotlib"],  # plot_tree
        "graphviz": ["graphviz", "matplotlib"],  # export_graphviz (matplotlib: images of the shapelets)
        "kmedoids": ["scikit-learn-extra"],  # Shapelets(selection='cluster')
        "profiling": ["line-profiler-pycharm"],  # RuleTree.utils.profiling
        "test": ["pytest", "numba", "graphviz", "matplotlib"],
        "all": ["numba", "graphviz", "matplotlib", "scikit-learn-extra"],
    },
    packages=find_packages(include=["RuleTree", "RuleTree.*"]),  # not benchmark and tests
)
