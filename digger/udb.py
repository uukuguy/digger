#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import leveldb
import pymongo
import six
import csv
import logging
import os
import time
from os import path
from logger import Logger, AppWatch
import msgpack
from utils import load_excel_to_rows, xldate_to_datetime

# ---------------- class Samples ----------------
class Samples():

    def __init__(self, samples_dir):
        self.samples_dir = samples_dir
        self.sample_id = 0
        self.db = None

        if not path.isdir(samples_dir):
            try:
                os.mkdir(samples_dir)
            except OSError:
                logging.error(Logger.error("mkdir %s failed." % (samples_dir)))
                return

        self.db = leveldb.LevelDB(self.samples_dir + "/content")

    def __del__(self):
        self.db = None

    # ---------------- acquire_sample_id() ----------------
    def acquire_sample_id(self, num_samples):
        sample_id = self.sample_id
        self.sample_id += num_samples
        return sample_id

    # ---------------- import_samples_from_xls() ----------------
    def import_samples_from_xls(self, xls_file):
        rows = load_excel_to_rows(xls_file)
        num_samples = len(rows)
        sample_id = self.acquire_sample_id(num_samples)

        batch_content = leveldb.WriteBatch()
        for row in rows:
            content = None
            if u"CONTENT" in row:
                content = row[u"CONTENT"]

            title = None
            if u"TITLE" in row:
                title = row[u"TITLE"]

            sample_data = (sample_id, title, content)
            rowstr = msgpack.dumps(sample_data)
            batch_content.Put(str(sample_id), rowstr)

            if sample_id % 100 == 0:
                logging.debug(Logger.debug("(%d/%d) %s" % (sample_id, num_samples, title)))
            sample_id += 1

        self.db.Write(batch_content, sync=True)

    def search_data(self, collection, begin_date, end_date):
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
            #result['url'] = item['url'].encode('utf-8')
            #result['publish_time'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item['result']['publish_time']))
            result['title'] = item['result'].get('title', '').encode('utf-8')
            result['text'] = item['result']['text'].encode('utf-8')
            #result['type'] = item['result']['type'].encode('utf-8')
            #result['keyword'] = item['result']['extra'].get('keyword', '').encode('utf-8')

            title = result['title']
            content = result['text']

            if not content.empty():
                sample_id = self.acquire_sample_id(1)
                if sample_id % 1000 == 0:
                    print sample_id, title
                sample_data = (sample_id, title, content)
                rowstr = msgpack.dumps(sample_data)
                self.db.Put(str(sample_id), rowstr)


    # ---------------- import_samples_from_mongodb() ----------------
    def import_samples_from_mongodb(self, mongodb, dbname, begin_date, end_date):
        mongo_client = pymongo.MongoClient(mongodb)
        #mongo_client = pymongo.MongoClient('139.196.189.136', 27017)
        db = mongo_client[dbname]
        coll_item = db.get_collection(dbname)
        self.search_data(coll_item, begin_date, end_date)

# ---------------- cmd_import_samples() ----------------
def cmd_import_samples(args):

    samples_name = None
    if hasattr(args, 'samples_name'):
        samples_name = args.samples_name

    xls_file = None
    if hasattr(args, 'xls_file'):
        xls_file = args.xls_file

    mongodb = None
    if hasattr(args, 'mongodb'):
        mongodb = args.mongodb

    dbname = None
    if hasattr(args, 'dbname'):
        dbname = args.dbname

    begin_date = None
    if hasattr(args, 'begin_date'):
        begin_date = args.begin_date

    end_date = None
    if hasattr(args, 'end_date'):
        end_date = args.end_date

    samples = Samples(samples_name)
    if xls_file is not None:
        samples.import_samples_from_xls(xls_file)
    if mongodb is not None:
        samples.import_samples_from_mongodb(mongodb, dbname, begin_date, end_date)

# ---------------- main() ----------------
def main():
    parser = argparse.ArgumentParser(description='Unstructured DataBase 0.1')

    subparsers = parser.add_subparsers(dest='subcommand', title='subcommands', help='sub-command help')

    # -------- import_samples --------
    parser_import_samples = subparsers.add_parser('import_samples', help='import samples help')
    parser_import_samples.add_argument('--samples_name', type=str, help='Samples name.')
    parser_import_samples.add_argument('--xls_file', type=str, help='The Excel file name will be imported.')
    parser_import_samples.add_argument('--mongodb', type=str, help='Mongodb mongodb://host:port/.')
    parser_import_samples.add_argument('--dbname', type=str, help='DB name in Mongodb.')
    parser_import_samples.add_argument('--begin_date', type=str, help='Start date.')
    parser_import_samples.add_argument('--end_date', type=str, help='End date.')

    parser_import_samples.set_defaults(func=cmd_import_samples)

    args = parser.parse_args()
    print args

    args.func(args)

if __name__ == '__main__':
    appwatch = AppWatch()
    main()
    appwatch.stop()

