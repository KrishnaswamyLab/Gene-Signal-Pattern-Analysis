import os
from setuptools import setup, find_packages

install_requires = [
    "graphtools>=1.5.0",
    "tensorflow>=2.6.0",
    "multiscale_phate==0.0",
    "numpy>=1.14.0",
    "scikit-learn",
    "scipy>=1.1.0",
    "tqdm",
    "scanpy",
    "phate"
]

test_requires = [
    "numpy>=1.14.0",
    "phate",
]

version_py = os.path.join(os.path.dirname(__file__), "gspa", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()

readme = open("README.md").read()

setup(
name='gspa',
version=version,
description="Gene Signal Pattern Analysis",
author='Aarthi Venkat, Krishnaswamy Lab, Yale University',
author_email='aarthi.venkat@yale.edu',
packages=find_packages(),
license="GNU General Public License Version 3",
install_requires=install_requires,
python_requires=">=3.6",
extras_require={"test": test_requires},
long_description=readme,
long_description_content_type="text/markdown",
url="https://github.com/KrishnaswamyLab/Gene-Signal-Pattern-Analysis",
download_url="https://github.com/KrishnaswamyLab/Gene-Signal-Pattern-Analysis/archive/v{}.tar.gz".format(version),
keywords=["big-data", "manifold-learning", "computational-biology", "dimensionality-reduction", "single-cell"],
classifiers=[
'Programming Language :: Python :: 3',
'Programming Language :: Python :: 3.6',
'Programming Language :: Python :: 3.7',
'Programming Language :: Python :: 3.8',
'Programming Language :: Python :: 3.9',
'Topic :: Scientific/Engineering :: Bio-Informatics',
'Intended Audience :: Developers',
'Intended Audience :: Science/Research',
'Natural Language :: English',
'Development Status :: 5 - Production/Stable',
'Operating System :: OS Independent',
],
)

setup_dir = os.path.dirname(os.path.realpath(__file__))
