from os.path import dirname, realpath
from setuptools import setup, find_packages, Distribution


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='OpenCOOD',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/cvims/TempCoBEV.git',
    license='Apache Licence 2.0',
    author='Dominik Roessle',
    author_email='dominik.roessle@thi.de',
    description='TempCoBEV',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)
