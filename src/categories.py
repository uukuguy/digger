#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
categories.py

'''

import bidict
import msgpack
import xlwt
import leveldb
import logging
from protocal import decode_sample_meta
from utils import sorted_dict


class Categories():
    # ---------------- __init__() ----------------
    def __init__(self, categories_dir):
        self.root_dir = categories_dir
        self.db_categories = None
        self.load_categories()


    # ---------------- __open_db_categories() ----------------
    def open_db_categories(self):
        if self.db_categories is None:
             self.db_categories = leveldb.LevelDB(self.root_dir)
        return self.db_categories

    # ---------------- __close_db_categories() ----------------
    def close_db_categories(self):
        self.db_categories = None

    # ---------------- load_a_categories() ----------------
    def load_a_categories(self, db_categories, categories_name):
        categories = bidict.bidict()
        try:
            str_categories = db_categories.Get(categories_name)
            dict_categories = msgpack.loads(str_categories)
            for k in dict_categories:
                category_id = dict_categories[k]
                k0 = k.decode('utf-8')
                categories[k0] = category_id
        except KeyError:
            categories = {}

        return categories

    # ---------------- load_categories() ----------------
    def load_categories(self):
        db_categories = self.open_db_categories()
        self.categories_1 = self.load_a_categories(db_categories, "__categories_1__")
        self.categories_2 = self.load_a_categories(db_categories, "__categories_2__")
        self.categories_3 = self.load_a_categories(db_categories, "__categories_3__")
        self.close_db_categories()

    # ---------------- save_a_categories() ----------------
    def save_a_categories(self, db_categories, categories, categories_name):
        dict_categories = {}
        for k in categories:
            category_id = categories[k]
            k0 = k.encode('utf-8')
            dict_categories[k0] = category_id
        db_categories.Put(categories_name, msgpack.dumps(dict_categories))

    # ---------------- save_categories() ----------------
    def save_categories(self):
        db_categories = self.open_db_categories()
        self.save_a_categories(db_categories, self.categories_1, "__categories_1__")
        self.save_a_categories(db_categories, self.categories_2, "__categories_2__")
        self.save_a_categories(db_categories, self.categories_3, "__categories_3__")
        self.close_db_categories()

    # ---------------- clear_categories() ----------------
    def clear_categories(self):
        self.categories_1 = bidict.bidict()
        self.categories_2 = bidict.bidict()
        self.categories_3 = bidict.bidict()

    # ---------------- get_category_name() ----------------
    def get_category_name(self, category_id):
        category_name = None
        if category_id % 1000 != 0:
            category_name = (~self.categories_3).get(category_id)
            #if category_name is None:
                #return self.get_category_name(int(category_id / 1000))
            #else:
                #return category_name.decode('utf-8')
        elif category_id % 1000000 != 0:
            category_name = (~self.categories_2).get(category_id)
            #if category_name is None:
                #return self.get_category_name(int(category_id / 1000))
            #else:
                #return category_name.decode('utf-8')
        else:
            category_name = (~self.categories_1).get(category_id)
            #if category_name is None:
                #return u""
            #else:
                #return category_name.decode('utf-8')
        if category_name is None:
            return u""
        else:
            return category_name

    # ---------------- get_categories_1_names() ----------------
    def get_categories_1_names(self):
        names_map = {}
        for category_name in self.categories_1:
            category_id = self.categories_1[category_name]
            names_map[category_id] = category_name
        names_list = sorted_dict(names_map)
        return [ cat_name for (cat_id, cat_name) in names_list]

    # ---------------- get_categories_1_idlist() ----------------
    def get_categories_1_idlist(self):
        idlist = []
        for category_name in self.categories_1:
            category_id = self.categories_1[category_name]
            idlist.append(category_id)
        return idlist


    # ---------------- get_category_1_id() ----------------
    @staticmethod
    def get_category_1_id(category_id):
        return int(category_id / 1000000) * 1000000

    # ---------------- get_category_2_id() ----------------
    @staticmethod
    def get_category_2_id(category_id):
        return int(category_id / 1000) * 1000

    # ---------------- get_category_1_name() ----------------
    @staticmethod
    def get_category_1_name(category_id):
        category_1_id = Categories.get_category_1_id(category_id)
        return get_category_name(category_1_id)

    # ---------------- get_category_2_name() ----------------
    @staticmethod
    def get_category_2_name(category_id):
        category_2_id = Categories.get_category_2_id(category_id)
        return get_category_name(category_2_id)

    # ---------------- print_categories() ----------------
    def print_categories(self):
        print "--------------- categories -----------------"
        categories_dict = {}
        for k in self.categories_1:
            v = self.categories_1[k]
            categories_dict[v] = k
        for k in self.categories_2:
            v = self.categories_2[k]
            categories_dict[v] = k
        for k in self.categories_3:
            v = self.categories_3[k]
            categories_dict[v] = k

        categories_list = sorted_dict(categories_dict)
        for (category_id, category_name) in categories_list:
            print "%s - %d" % (category_name, category_id)

    # ---------------- get_category_id() ----------------
    def get_category_id(self, cat1, cat2 = None, cat3 = None):
        if not cat3 is None:
            category_name = u"%s:%s:%s:" % (cat1, cat2, cat3)
            return self.categories_3.get(category_name)
        elif not cat2 is None:
            category_name = u"%s:%s::" % (cat1, cat2)
            return self.categories_2.get(category_name)
        elif not cat1 is None:
            category_name = u"%s:::" % (cat1)
            return self.category_1.get(category_name)
        else:
            return None

    def create_or_get_category_id(self, cat1, cat2, cat3):
        print_msg = False
        if cat1 == u"依法治企" and cat2 == u"供电服务":
            print_msg = True

        category = -1
        if cat1 != u"":
            #print cat1.__class__, cat1
            category_1_text = u"%s:::" % (cat1)
            category_1 = self.categories_1.setdefault(category_1_text, (len(self.categories_1) + 1) * 1000000)
            category = category_1
            if print_msg:
                print u"cat_1 %d" % (category_1)
            if cat2 != "":
                category_2_text = u"%s:%s::" % (cat1, cat2)
                category_2 = self.categories_2.setdefault(category_2_text, category_1 + (len(self.categories_2) + 1) * 1000)
                category = category_2
                if print_msg:
                    print u"cat_2 %d" % (category_2)
                if cat3 != "":
                    category_3_text = u"%s:%s:%s:" % (cat1, cat2, cat3)
                    category_3 = self.categories_3.setdefault(category_3_text, category_2 + len(self.categories_3) + 1)
                    category = category_3
                    if print_msg:
                        print u"cat_3 %d" % (category_3)

        if print_msg:
            print u"%s %s %d" % (cat1, cat2, category)
        return category


    # ---------------- export_categories_to_xls() ----------------
    def export_categories_to_xls(self, categories_useinfo, xls_file):
        categories = self

        if xls_file is None:
            return

        wb = xlwt.Workbook(encoding='utf-8')
        ws = wb.add_sheet("categories")

        ws.write(0, 0, 'ID')
        ws.write(0, 1, 'CAT1')
        ws.write(0, 2, 'CAT2')
        ws.write(0, 3, 'CAT3')
        ws.write(0, 4, 'SAMPLES')

        rowidx = 1
        categories_list = sorted_dict(categories_useinfo)
        for (category_id, category_used) in categories_list:
            if category_id % 1000 == 0:
                if category_id % 1000000 == 0:
                    category_3 = -1
                    category_2 = -1
                    category_1 = category_id
                else:
                    category_3 = -1
                    category_2 = category_id
                    category_1 = int(category_id / 1000000) * 1000000
            else:
                category_3 = category_id
                category_2 = int(category_id / 1000) * 1000
                category_1 = int(category_id / 1000000) * 1000000

            category_1_name = categories.get_category_name(category_1)
            category_2_name = categories.get_category_name(category_2)
            category_3_name = categories.get_category_name(category_3)

            logging.debug("id:%d 1:%d 2:%d 3:%d" % ( category_id, category_1, category_2, category_3))

            ws.write(rowidx, 0, category_id)
            ws.write(rowidx, 1, category_1_name)
            ws.write(rowidx, 2, category_2_name)
            ws.write(rowidx, 3, category_3_name)
            ws.write(rowidx, 4, category_used)

            rowidx += 1

        wb.save(xls_file)
        logging.debug("Export categories to xls file %s" % (xls_file) )


