#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
categories.py

'''

import bidict
import msgpack
from protocal import decode_sample_meta
from utils import sorted_dict

class Categories():
    # ---------------- __init__() ----------------
    def __init__(self, db_content):
        self.db_content = db_content
        self.load_categories()


    # ---------------- load_a_categories() ----------------
    def load_a_categories(self, categories_name):
        categories = bidict.bidict()
        try:
            str_categories = self.db_content.Get(categories_name)
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
        self.categories_1 = self.load_a_categories("__categories_1__")
        self.categories_2 = self.load_a_categories("__categories_2__")
        self.categories_3 = self.load_a_categories("__categories_3__")

    # ---------------- save_a_categories() ----------------
    def save_a_categories(self, categories, categories_name):
        dict_categories = {}
        for k in categories:
            category_id = categories[k]
            k0 = k.encode('utf-8')
            dict_categories[k0] = category_id
        self.db_content.Put(categories_name, msgpack.dumps(dict_categories))

    # ---------------- save_categories() ----------------
    def save_categories(self):
        self.save_a_categories(self.categories_1, "__categories_1__")
        self.save_a_categories(self.categories_2, "__categories_2__")
        self.save_a_categories(self.categories_3, "__categories_3__")

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

    # ---------------- get_category_id_1() ----------------
    def get_category_id_1(self, category_id):
        return int(category_id / 1000000) * 1000000

    # ---------------- get_category_id_2() ----------------
    def get_category_id_2(self, category_id):
        return int(category_id / 1000) * 1000

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
            category_name = "%s:%s:%s:" % (cat1, cat2, cat3)
            return self.categories_3.get(category_name)
        elif not cat2 is None:
            category_name = "%s:%s::" % (cat1, cat2)
            return self.categories_2.get(category_name)
        elif not cat1 is None:
            category_name = "%s:::" % (cat1)
            return self.category_1.get(category_name)
        else:
            return None

    def build_category_id(self, cat1, cat2, cat3):
        print_msg = False
        if cat1 == u"电力改革" and cat2 == u"三集五大":
            print_msg = False

        category = -1
        if cat1 != "":
            category_1_text = "%s:::" % (cat1)
            category_1 = self.categories_1.setdefault(category_1_text, (len(self.categories_1) + 1) * 1000000)
            category = category_1
            if print_msg:
                print "cat_1 %d" % (category_1)
            if cat2 != "":
                category_2_text = "%s:%s::" % (cat1, cat2)
                category_2 = self.categories_2.setdefault(category_2_text, category_1 + (len(self.categories_2) + 1) * 1000)
                category = category_2
                if print_msg:
                    print "cat_2 %d" % (category_2)
                if cat3 != "":
                    category_3_text = "%s:%s:%s:" % (cat1, cat2, cat3)
                    category_3 = self.categories_3.setdefault(category_3_text, category_2 + len(self.categories_3) + 1)
                    category = category_3
                    if print_msg:
                        print "cat_3 %d" % (category_3)

        if print_msg:
            print "%s %s %d" % (cat1, cat2, category)
        return category


    # ---------------- rebuild_categories() ----------------
    def rebuild_categories(self):

        self.clear_categories()

        batch_content = leveldb.WriteBatch()
        rowidx = 0
        for i in self.db_content.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                continue
            (sample_id, category, date, title, key, url, msgext) = decode_sample_meta(i[1])
            (version, content, (cat1, cat2, cat3)) = msgext
            #try:
                #(version, content, (cat1, cat2, cat3)) = msgext
            #except ValueError:
                #bad_samples.append(sample_id)
                #rowidx += 1
                #continue

            version = "1"
            msgext = (version, content, (cat1, cat2, cat3))

            category_id = self.build_category_id(cat1, cat2, cat3)

            sample_data = (sample_id, category_id, date, title, key, url, msgext)
            rowstr = msgpack.dumps(sample_data)
            batch_content.Put(str(sample_id), rowstr)

            #logging.debug("[%d] %d %d=<%s:%s:%s:>" % (rowidx, sample_id, category_id, cat1, cat2, cat3))

            rowidx += 1

        #logging.debug("Delete %d bad samples." % (len(bad_samples)))
        #for sample_id in bad_samples:
            #self.db_content.Delete(str(sample_id))

        self.db_content.Write(batch_content, sync=True)
        self.save_categories()
        self.print_categories()


    # ---------------- get_categories_list() ----------------
    def get_categories_list(self):
        categories = {}
        for category_1 in (~self.categories_1):
            categories[category_1] = 0
        for category_2 in (~self.categories_2):
            categories[category_2] = 0
        for category_3 in (~self.categories_3):
            categories[category_3] = 0

        unknown_categories = {}

        rowidx = 0
        for i in self.db_content.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                continue

            (sample_id, category, date, title, key, url, msgext) = decode_sample_meta(i[1])
            (version, content, (cat1, cat2, cat3)) = msgext

            if not category in categories:
                if category in unknown_categories:
                    unknown_categories[category] += 1
                else:
                    unknown_categories[category] = 1
            else:
                categories[category] += 1

            rowidx += 1

        return categories

    # ---------------- print_categories_info() ----------------
    def print_categories_info(self, categories):
        categories_list = sorted_dict(categories)
        f = open("./result/categories.txt", 'wb+')
        for (category_id, category_used) in categories_list:
            category_name = self.get_category_name(category_id)
            str_category = "%d - %s %d samples" % (category_id, category_name, category_used)
            print str_category
            f.write("%s\n" % (str_category.encode('utf-8')))

        f.close()

        #print "%d unknown categories" % (len(unknown_categories))
        #for category_id in unknown_categories:
            #category_name = self.get_category_name(category_id)
            #category_used = unknown_categories[category_id]
            #print "<Unknown> %d - %s %d samples." % (category_id, category_name , category_used)


    # ---------------- export_categories_to_xls() ----------------
    def export_categories_to_xls(self, categories, xls_file):
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
        categories_list = sorted_dict(categories)
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

            category_1_name = self.get_category_name(category_1)
            category_2_name = self.get_category_name(category_2)
            category_3_name = self.get_category_name(category_3)

            logging.debug("id:%d 1:%d 2:%d 3:%d" % ( category_id, category_1, category_2, category_3))

            ws.write(rowidx, 0, category_id)
            ws.write(rowidx, 1, category_1_name)
            ws.write(rowidx, 2, category_2_name)
            ws.write(rowidx, 3, category_3_name)
            ws.write(rowidx, 4, category_used)

            rowidx += 1

        wb.save(xls_file)
        logging.debug("Export categories to xls file %s" % (xls_file) )

