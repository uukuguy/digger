#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Digger
'''

import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")

application = tornado.web.Application([
    (r"/", MainHandler),
    ])

if __name__ == '__main__':
    application.listen(2015)
    tornado.ioloop.IOLoop.instance().start()



