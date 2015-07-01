#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division
import random
import logging
import math

logging.basicConfig(
        level = logging.DEBUG,
        format = "[%(asctime)s] %(name)s:%(levelname)s: %(message)s"
        )

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
def open_excel(file):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception, e:
        print str(e)

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
            logging.debug("Save svm: %d/%d" % (idx, len(X)))
        idx += 1

    file_out.close()



