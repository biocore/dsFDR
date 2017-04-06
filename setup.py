#!/usr/bin/env python

from setuptools import find_packages, setup
import dsfdr


classifiers = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: BSD License',
    'Environment :: Console',
    'Topic :: Software Development :: Libraries',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Operating System :: Unix',
    'Operating System :: POSIX',
    'Operating System :: Microsoft',
    'Operating System :: MacOS :: MacOS X']


description = 'descrete False Discovery Rate method'
with open('README.md') as f:
    long_description = f.read()


setup(name='dsfdr',
      version=dsfdr.__version__,
      license='BSD',
      description=description,
      long_description=long_description,
      classifiers=classifiers,
      keywords='statistics FDR',
      author="",
      packages=find_packages(),
      author_email="",
      maintainer_email="",
      url='http://github.com/biocore/dsFDR',
      test_suite='nose.collector',
      install_requires=['numpy', 'scipy'],
      extras_require={'test': ["nose", "pep8", "flake8"],
                      'coverage': ["coverage"],
                      'doc': ["Sphinx"]})
