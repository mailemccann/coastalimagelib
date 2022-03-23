 from setuptools import setup
 import conda_build.bdist_conda

setup(
    name="coastalimagelib",
    version="v1.0.0",
    distclass=conda_build.bdist_conda.CondaDistribution,
    conda_buildnum=1,
)