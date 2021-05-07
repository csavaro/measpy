from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Measurements with Python'
LONG_DESCRIPTION = 'Calsses and methods to do data acquisition and processing'

# Setting up
setup(
    name="measpy",
    version=VERSION,
    author="Olivier Doaré",
    author_email="<olivier.doare@ensta-paris.fr>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy','matplotlib'],
    keywords=['Python', 'Measurements', 'Data acquisition'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3"
        ]
)