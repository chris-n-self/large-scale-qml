from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
#     long_description = f.read()

setup(
    name='largescaleqml',
    version='0.0.1',
    description='',
    long_description='',
    url='https://github.com/chris-n-self/large-scale-qml',
    author='Chris Self, Tobias Haug',
    classifiers=[],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests']),

    install_requires=[
        'qcoptim',
    ],
    extras_require={  # Optional
    },

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/chris-n-self/large-scale-qml/issues',
        'Source': 'https://github.com/chris-n-self/large-scale-qml/',
    },
)
