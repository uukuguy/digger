#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
transform.py -
'''

import leveldb
import logging
from logger import Logger
import msgpack
import xlwt
from utils import load_excel_to_rows
from categories import Categories
from protocal import decode_sample_meta

# ---------------- import_samples_from_xls() ----------------
def import_samples_from_xls(samples, categories, xls_file):
    corpus = samples.corpus

    rows = load_excel_to_rows(xls_file)
    num_samples = len(rows)
    sample_id = corpus.acquire_sample_id(num_samples)

    batch_content = leveldb.WriteBatch()
    for row in rows:
        category = -1
        if u"CATEGORY" in row:
            category = int(row[u"CATEGORY"])

        content = row[u"CONTENT"]
        if content == "":
            content = None

        title = ""
        if u"TITLE" in row:
            title = row[u"TITLE"]

        date = ""
        if u"DATE" in row:
            row_date = row[u"DATE"]
            if row_date.__class__ is str:
                date = row_date.decode('utf-8')
            else:
                date = str(row_date).decode('utf-8')

        key = ""
        if u"KEY" in row:
            key = row[u"KEY"]
            if key.__class__ != str:
                key = str(key).decode('utf-8')

        url = ""
        if u"URL" in row:
            url = row[u"URL"]

        cat1 = ""
        if u"CAT1" in row:
            cat1 = row[u"CAT1"].strip()

        cat2 = ""
        if u"CAT2" in row:
            cat2 = row[u"CAT2"].strip()

        cat3 = ""
        if u"CAT3" in row:
            cat3 = row[u"CAT3"].strip()

        version = "1"
        msgext = (version, content, (cat1, cat2, cat3))

        category_id = categories.create_or_get_category_id(cat1, cat2, cat3)

        sample_data = (sample_id, category_id, date, title, key, url, msgext)
        rowstr = msgpack.dumps(sample_data)
        batch_content.Put(str(sample_id), rowstr)

        if sample_id % 100 == 0:
            logging.debug(Logger.debug("Row: %d/%d %s %s" % (sample_id, len(rows), date, title)))
        sample_id += 1

    return batch_content


# ---------------- export_samples_to_xls() ----------------
def export_samples_to_xls(samples, xls_file):

    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet("negative opinions")
    ws.write(0, 0, 'CATEGORY')
    ws.write(0, 1, 'DATE')
    ws.write(0, 2, 'CAT1')
    ws.write(0, 3, 'CAT2')
    ws.write(0, 4, 'TITLE')
    ws.write(0, 5, 'KEY')
    ws.write(0, 6, 'URL')
    ws.write(0, 7, 'CONTENT')

    rowidx = 0
    for i in samples.db_content.RangeIter():
        row_id = i[0]
        if row_id.startswith("__"):
            continue
        (sample_id, category, date, title, key, url, msgext) = decode_sample_meta(i[1])
        (version, content, (cat1, cat2, cat3)) = msgext
        if content is None:
            content = ""
        if len(content) >= 1024 * 32:
            content = content[:1024*32 - 1]
        ws.write(rowidx + 1, 0, category)
        ws.write(rowidx + 1, 1, date)
        ws.write(rowidx + 1, 2, cat1)
        ws.write(rowidx + 1, 3, cat2)
        ws.write(rowidx + 1, 4, title)
        ws.write(rowidx + 1, 5, key)
        ws.write(rowidx + 1, 6, url)
        ws.write(rowidx + 1, 7, content)

        if rowidx % 100 == 0:
            logging.debug(Logger.debug("[%d] %d %s" % (rowidx, sample_id, title)))
        rowidx += 1

    wb.save(xls_file)



# ---------------- export_urls_to_xls() ----------------
def export_urls_to_xls(xls_file, none_samples, empty_samples, normal_samples):

    wb = xlwt.Workbook(encoding='utf-8')
    ws = wb.add_sheet("urls")
    ws.write(0, 0, 'url')
    ws.write(0, 1, 'status')
    rowidx = 1
    for (sample_id, url) in none_samples:
        if len(url) > 0:
            ws.write(rowidx, 0, url)
            ws.write(rowidx, 1, "None")
            rowidx += 1

    for (sample_id, url) in empty_samples:
        if len(url) > 0:
            ws.write(rowidx, 0, url)
            ws.write(rowidx, 1, "Empty")
            rowidx += 1

    for (sample_id, url) in normal_samples:
        if len(url) > 0:
            ws.write(rowidx, 0, url)
            ws.write(rowidx, 1, "Normal")
            rowidx += 1

    wb.save(xls_file)

