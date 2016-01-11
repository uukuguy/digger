#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import leveldb
import logging
import os
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

# ---------------- cmd_import_samples() ----------------
def cmd_import_samples(args):

    xls_file = None
    if hasattr(args, 'xls_file'):
        xls_file = args.xls_file
    samples_name = None
    if hasattr(args, 'samples_name'):
        samples_name = args.samples_name

    samples = Samples(samples_name)
    samples.import_samples_from_xls(xls_file)


# ---------------- main() ----------------
def main():
    parser = argparse.ArgumentParser(description='Unstructured DataBase 0.1')

    subparsers = parser.add_subparsers(dest='subcommand', title='subcommands', help='sub-command help')

    # -------- import_samples --------
    parser_import_samples = subparsers.add_parser('import_samples', help='import samples help')
    parser_import_samples.add_argument('--xls_file', type=str, help='The Excel file name will be imported.')
    parser_import_samples.add_argument('--samples_name', type=str, help='Samples name.')
    parser_import_samples.set_defaults(func=cmd_import_samples)

    args = parser.parse_args()
    print args

    args.func(args)

if __name__ == '__main__':
    appwatch = AppWatch()
    main()
    appwatch.stop()

