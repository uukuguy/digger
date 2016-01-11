#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# vim: set et sw=4 ts=4 sts=4 ff=unix fenc=utf8:
# Author: chenqian
import pymongo
import sys
import time
import csv
import click
import logging
import six

logging.basicConfig( \
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
    filename='search.log',
    filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def parse_result(result, fieldnames):
    """
    删除不需要保存的字段
    """
    # res= [del result[key] for key in result.keys() if key not in fieldnames]
    if fieldnames:
        for key in result.keys():
            if key not in fieldnames:
                del result[key]


def search_data(collection, fieldnames, begin_date, end_date):
    """
    根据条件查询
    """
    nbegin_date = int(time.mktime(map(eval, "{}-00-00-00-0-0-0".format(begin_date).split('-'))))
    nend_date = int(time.mktime(map(eval, "{}-23-59-59-0-0-0".format(end_date).split('-'))))
    condition = {'result.publish_time': {'$gte': nbegin_date, '$lte': nend_date}}
    logging.info('search: {},{}'.format(collection.name, condition))
    result = {}
    # for item in collection.find(condition).sort('result.publish_time', pymongo.ASCENDING):
    for item in collection.find(condition):
        result['url'] = item['url'].encode('utf-8')
        result['publish_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item['result']['publish_time']))
        result['title'] = item['result'].get('title', '').encode('utf-8')
        result['text'] = item['result']['text'].encode('utf-8')
        result['type'] = item['result']['type'].encode('utf-8')
        result['keyword'] = item['result']['extra'].get('keyword', '').encode('utf-8')

        # 去除不用的字段
        parse_result(result, fieldnames)
        yield result


def write_to_csv(result, fieldnames, filename):
    """
    保存到csv
    """
    with open('./data/' + filename, 'w') as f:
        f_csv = csv.DictWriter(f, fieldnames)
        for item in result:
            f_csv.writerow(item)
    f.close()


def read_config(ctx, param, value):
    """
    读取配置文件
    """
    if not value:
        return {}
    import json

    def underline_dict(d):
        if not isinstance(d, dict):
            return d
        return dict((k.replace('-', '_'), underline_dict(v)) for k, v in six.iteritems(d))

    config = underline_dict(json.load(value))
    ctx.default_map = config
    return config


@click.command()
@click.option('-c', '--config', callback=read_config, type=click.File('r'), help=u'json 配置文件')
@click.option('--host', default='139.196.189.136', help=u'阿里云ip.')
@click.option('--port', default=27017, help=u'mongodb 端口')
@click.option('--begin_date', help=u'起始时间')
@click.option('--end_date', help=u'结束时间')
@click.option('--fieldnames', default='url,publish_time,title,text,type,keyword', help=u'查询的结果')
@click.option('--search_collection', default='all', help=u'查询的collection名字')
def cli(**kawrgs):
    #conn = pymongo.Connection(kawrgs['host'], kawrgs['port'])
    conn = pymongo.MongoClient(kawrgs['host'], kawrgs['port'])
    #conn = pymongo.MongoClient("mongodb://139.196.189.136:27017/")
    #conn = pymongo.MongoClient("139.196.189.136", 27017)
    db = conn['resultdb']
    if kawrgs['search_collection'] != 'all':
        for coll_item_name in kawrgs['search_collection'].split(','):
            coll_item = db.get_collection(coll_item_name)
            data = search_data(coll_item, kawrgs['fieldnames'], kawrgs['begin_date'], kawrgs['end_date'])
            write_to_csv(data, kawrgs['fieldnames'].split(','), coll_item_name + '.csv')
    else:
        for coll_item_name in db.collection_names():
            coll_item = db.get_collection(coll_item_name)

            data = search_data(coll_item, kawrgs['fieldnames'], kawrgs['begin_date'], kawrgs['end_date'])
            write_to_csv(data, kawrgs['fieldnames'].split(','), coll_item_name + '.csv')

    logging.info("ok")


if __name__ == '__main__':
    cli()
