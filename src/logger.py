#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
logger.py
'''

#import logging

class Logger():
    def __init__(self):
        pass

    @staticmethod
    def trace(msg):
        return "\033[1;33m%s\033[0m" % msg

    @staticmethod
    def debug(msg):
        return "\033[1;34m%s\033[0m" % msg

    @staticmethod
    def info(msg):
        return "\033[1;32m%s\033[0m" % msg

    @staticmethod
    def notice(msg):
        return "\033[1;36m%s\033[0m" % msg

    @staticmethod
    def warn(msg):
        return "\033[1;35m%s\033[0m" % msg

    @staticmethod
    def error(msg):
        return "\033[1;31m%s\033[0m" % msg

