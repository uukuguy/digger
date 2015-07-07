#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
category_feature_matrix.py - 分类特征矩阵

CategoryFeatureMatrix.samples
    {category_id, ({feature_id:feature_weight})}
    feature_weight: (pd_word, specialty, popularity)

'''

import leveldb
import msgpack

class CategoryFeatureMatrix():

    # ---------------- __init__() ----------------
    def __init__(self):
        self.cf_matrix = {}

    # ---------------- get_features() ----------------
    def get_features(self, category_id):
        if category_id in self.cf_matrix:
            return self.cf_matrix[category_id]
        else:
            return None

    # ---------------- set_features() ----------------
    def set_features(self, category_id, features_info):
        #print features_info
        self.cf_matrix[category_id] = features_info

    # ---------------- clear() ----------------
    def clear(self):
        self.cf_matrix = {}

    # ---------------- save() ----------------
    def save(self, file_cfm):
        #print len(self.cf_matrix)
        #for category_id in self.cf_matrix:
            #features_info = self.cf_matrix[category_id]
            #print features_info
        f = open(file_cfm, "wb+")
        str_cfm = msgpack.dumps(self.cf_matrix)
        f.write(str_cfm)
        f.close()

    # ---------------- load() ----------------
    def load(self, file_cfm):
        self.clear()
        f = open(file_cfm, "r")
        str_cfm = f.read()
        self.cf_matrix = msgpack.loads(str_cfm)
        f.close()



