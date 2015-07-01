#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
protocal.py - 序列化协议
'''
import msgpack

def decode_sample_meta(str_sample_meta):
    (sample_id, category, date, title, key, url, msgext) = msgpack.loads(str_sample_meta)
    version = u""
    cat1 = u""
    cat2 = u""
    cat3 = u""
    if msgext.__class__ is str:
        content = msgext
    else:
        (version, content, extdata) = msgext
        if version == "1":
            (cat1, cat2, cat3) = extdata
    if not content is None:
        content = content.decode('utf-8')
    msgext = (version.decode('utf-8'), content, (cat1.decode('utf-8'), cat2.decode('utf-8'), cat3.decode('utf-8')))

    return (sample_id, category, date.decode('utf-8'), title.decode('utf-8'), key.decode('utf-8'), url.decode('utf-8'), msgext)
