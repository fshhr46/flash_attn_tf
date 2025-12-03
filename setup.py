"""TensorFlow Snap Addons.

Tensorflow Snap Addons is a collection of custom Tensorflow extensions broadly used at Snap Inc.
"""

import datetime
import sys
from pathlib import Path

from setuptools import find_namespace_packages, setup
from setuptools.dist import Distribution

_DOCLINES = __doc__.split("\n")
_PROJECT_NAME = "flash_attn_tf"
_PROJECT_VERSION = "0.1.0"


def _get_utcnow_str():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")


def _get_project_version():
    if "--release-version" in sys.argv:
        sys.argv.remove("--release-version")
        return _PROJECT_VERSION
    return f"{_PROJECT_VERSION}-dev{_get_utcnow_str()}"


class _BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return True


setup(
    name=_PROJECT_NAME,
    version=_get_project_version(),
    description=_DOCLINES[0],
    long_description="\n".join(_DOCLINES[2:]),
    author="Haoran Huang",
    author_email="fshhr46@gmail.com",
    packages=find_namespace_packages(include=["flash_attn_tf*"]),
    install_requires=Path("requirements.txt").read_text().splitlines(),
    include_package_data=True,
    zip_safe=False,
    distclass=_BinaryDistribution,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    keywords="tensorflow snap addons machine learning",
)
