#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division
import random
import logging
from os import path
from logger import Logger
import math
from datetime import datetime

def is_chinese_word(u0):
    return u0 >= u'\u4e00' and u0 <= u'\u9fa5'

def to_chinese_string(objects):
    return str(objects).replace('u\'', '\'').decode("unicode-escape")

def sorted_dict(thedict, reverse=False):
    return [(k, thedict[k]) for k in sorted(thedict.keys(), reverse=reverse)]

def sorted_dict_by_values(thedict, reverse=False):
    return sorted(thedict.iteritems(), key=lambda d:d[1], reverse = reverse)

def good_dict_string(thedict, reverse=False):
    return to_chinese_string(sorted_dict(thedict, reverse=reverse))

# ================ Excel Reader ================
import xlrd, xlwt, xlsxwriter

# ---------------- open_excel() ----------------
def open_excel(file):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception, e:
        print str(e)


# ---------------- xldate_to_datetime() ----------------
def xldate_to_datetime(value):
    (y, m, d, h, mi, s) = xlrd.xldate.xldate_as_tuple(value, 1)
    return datetime(y, m, d, h, mi, s)

# ---------------- load_excel_to_rows() ----------------
def load_excel_to_rows(xls_file, sheet_idx = 0, colname_idx = 0):
    rows = []

    data = open_excel(xls_file)
    if data is not None:
        table = data.sheets()[sheet_idx]
        nrows = table.nrows
        ncols = table.ncols
        print "xls_file: " + xls_file + " rows: " + str(nrows) + " cols: " +str(ncols)

        colnames = table.row_values(colname_idx)
        print to_chinese_string(colnames)

        for rownum in range(1, nrows):
            values = table.row_values(rownum)
            if values:
                row = {}
                for i in range(len(colnames)):
                    row[colnames[i]] = values[i]
                row[u"ID"] = rownum
                rows.append(row)

    random_rows = []
    while len(rows) > 0:
        idx = random.randint(0, len(rows) - 1)

        random_rows.append(rows[idx])

        del rows[idx]

    print "Load %s Done." % (xls_file)
    return random_rows


# ---------------- save_as_svm_file() ----------------
def save_as_svm_file(f, X, y):
    if f.__class__ is file:
        file_out = f
    else:
        file_out = open(svm_file, "wb+")

    idx = 0
    for x in X:
        category = y[idx]
        f.write("%d " % (category))

        x_list = sorted_dict(x)
        for (key, value) in x_list:
            if type(value) is float:
                f.write("%d:%.6f " % (key, value))
            else:
                f.write("%d:%d " % (key, value))
        file_out.write("\n")
        if idx % 1000 == 0:
            logging.debug(Logger.debug("Save svm: %d/%d" % (idx, len(X))))
        idx += 1

    file_out.close()


# ---------------- crossvalidation_list_by_ratio() ----------------
def crossvalidation_list_by_ratio(X, ratio):
    num = len(X)
    if num == 0:
        return [], []
    if ratio == 1.0:
        return X, []
    num1 = int(num * ratio + 0.5)

    X1 = []
    X2 = [a for a in X]
    while len(X1) < num1:
        idx = random.randint(0, len(X2) - 1)
        X1.append(X2[idx])
        del X2[idx]

    return X1, X2


import leveldb, msgpack

# ---------------- save_objlist() ----------------
def save_objlist(self, db_path, objlist):
    db = leveldb.LevelDB(db_path)
    wb = db.WriteBatch()
    n = 0
    for obj in self.objlist:
        wb.Put(str(n), msgpack.dumps(obj.dumps()))
        n += 1
    db.Write(wb, sync=True)
    db = None


# ---------------- load_objlist() ----------------
def load_objlist(self, db_path, objlist, Obj):
    objlist.clear()

    db = leveldb.LevelDB(db_path)
    for i in db.RangeIter():
        row_id = i[0]
        if row_id[0:2] == "__":
            continue
        obj = Obj.loads(i[i])
        objlist.append(obj)
    db = None


from ConfigParser import ConfigParser

class AppArgs():
    def __init__(self, conf_files = None):

        # {section:{option:value}}
        self.args_map = {}

        if not conf_files is None:
            for conf_file in conf_files:
                self.parse_from_file(conf_file)


    def parse_from_file(self, file_name):
        if not path.isfile(file_name):
            return
        logging.info(Logger.notice('AppArgs parse %s' % (path.abspath(file_name))))

        conf = ConfigParser()
        conf.read(file_name)
        for section in conf.sections():
            for (option, value) in conf.items(section):
                self.set_arg(option, option, value)

    def get_arg(self, section, option):
        if section in self.args_map:
            options = self.args_map[section]
            if option in options:
                return options[option]
        return None

    def set_arg(self, section, option, value):
        if section in self.args_map:
            options = self.args_map[section]
        else:
            options = {}
        options[option] = value
        self.args_map[section] = options

    def update_arg(self, option, value, section='global'):
        if not value is None:
            self.set_arg(section, option, value)

    def write_to_file(self, file_name):
        writed = False
        conf = ConfigParser()

        for section in self.args_map:
            conf.add_section(section)
            options = self.args_map[section]
            for option in options:
                value = options[option]
                conf.set(section, option, value)

        fp = open(file_name, 'wb+')
        conf.write(fp)
        fp.close()

        logging.info(Logger.notice('AppArgs write %s' % (file_name)))


    def print_args(self):
        for section in self.args_map:
            logging.info(Logger.notice("Section: %s" % (section)))
            options = self.args_map[section]
            for option in options:
                value = options[option]
                logging.info(Logger.notice("Opinion %s:%s" % (option, value)))


