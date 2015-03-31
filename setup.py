import os
from setuptools import find_packages
from setuptools import setup

version = '0.1dev'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
    CHANGES = ''
    #CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()
except IOError:
    README = CHANGES = ''

install_requires = [
    'numpy',
    'Theano',
    'Lasagne'
    ]

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    ]

docs_require = [
    'Sphinx',
    ]

setup(
    name="DeepModels",
    version=version,
    description="Deep models for theano",
    long_description="\n\n".join([README, CHANGES]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        ],
    keywords="",
    author="Soren Sonderby",
    author_email="skaaesonderby@gmail.com",
    url="https://github.com/skaae/lasagne-draw/",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        'docs': docs_require,
        },
    )
