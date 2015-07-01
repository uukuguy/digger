#!/usr/bin/env python
# -*- coding:utf-8 -*-
#from  test1 import Test
import test1

class Test():
    def __init__(self):
        self.name = 'a'

    def output(self):
        print self.name

if __name__ == '__main__':
    test = Test()
    test.output()
    test1 = test1.Test()
    test1.show()
