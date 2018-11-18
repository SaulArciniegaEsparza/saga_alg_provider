"""
SAGA GIS algorithm provider
Installation file
"""

from setuptools import setup, find_packages

setup(
    name='pysaga',
    version='1.0.0',
    author='Saul Arciniega Esparza',
    author_email='zaul.ae@gmail.com',
    description='A SAGA GIS algorithm provider',
    license='BSD',
    packages=find_packages(exclude=['build', 'dist', 'pysaga.egg-info']),
    classifiers=[
        'Development Status :: SAGA GIS algorithm provider',
        'Intended Audience :: Engineering software',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.6'],
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'numba']
)
