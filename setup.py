#!/usr/bin/env python
# coding: UTF-8


from distutils.core import setup

setup(
	name         = 'padexp',
	version      = 0.1,
	description  = 'Padé approximation of exponential functions',
	author = 'Olivier Verdier',
	url = 'https://github.com/olivierverdier/padexp',
	license      = 'MIT',
	keywords = ['Math', 'Phi function', 'Matrix exponential',],
	packages=['padexp',],
	classifiers = [
	'Development Status :: 4 - Beta',
	'Intended Audience :: Science/Research',
	'License :: OSI Approved :: BSD License',
	'Operating System :: OS Independent',
	'Programming Language :: Python',
	'Topic :: Scientific/Engineering :: Mathematics',
	],
	)
