from setuptools import setup

setup(
    name='pysaga',
    version='1.0.0',
    author='Saul Arciniega Esparza',
    author_email='zaul.ae@gmail.com',
    description='A SAGA GIS algorithm provider',
    license='BSD',
    packages=['pysaga'],
    classifiers=[
        'Development Status :: SAGA GIS algorithm provider',
        'Intended Audience :: Engineering software',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules'],
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      '']
)
