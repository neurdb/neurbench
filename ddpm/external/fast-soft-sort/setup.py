# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Install fast_soft_sort."""

from setuptools import find_packages, setup

setup(
    name="fast_soft_sort",
    version="0.1",
    description=("Differentiable sorting and ranking in O(n log n)."),
    author="Google LLC",
    author_email="no-reply@google.com",
    url="https://github.com/google-research/fast-soft-sort",
    license="BSD",
    packages=find_packages(),
    package_data={},
    install_requires=[
        "numba",
        "numpy",
        "scipy>=1.2.0",
    ],
    extras_require={
        "tf": ["tensorflow>=1.12"],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine learning sorting ranking",
)
