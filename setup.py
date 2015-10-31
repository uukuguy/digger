#!/usr/bin/env python
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages
setup(name='Digger',
        version='1.0',
        description='Machine Learning Framework.',
        author='Jiangwen Su',
        author_email='uukuguy@gmail.com',
        url='',
        packages=find_packages(),
        scripts=['scripts/digger'],
        install_requires=['leveldb', 'msgpack-python', 'jieba']
        )

