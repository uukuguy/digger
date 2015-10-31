#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
logger.py
'''

import logging

logging.basicConfig(
        level = logging.DEBUG,
        format = "[%(asctime)s %(relativeCreated)d] %(filename)s(%(lineno)d) %(module)s.%(funcName)s() %(name)s:%(levelname)s: %(message)s"
        )

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

from datetime import datetime
class AppWatch():
    def __init__(self):
        self.s_time = datetime.utcnow()

    def stop(self):
        e_time = datetime.utcnow()
        t_time = (e_time - self.s_time)
        logging.info(Logger.notice("Done.(%s)" % (str(t_time))))

