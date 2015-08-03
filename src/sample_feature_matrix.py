#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
sample_feature_matrix.py - 样本特征矩阵

SampleFeatureMatrix.samples
    {sample_id, (category_id, {feature_idx:feature_weight})}

'''

from scipy.sparse import csr_matrix
import numpy as np
import bidict
import msgpack
import tokenize
import StringIO
from utils import sorted_dict
import logging
from logger import Logger

class SampleFeatureMatrix():
    def __init__(self, category_id_map = None, feature_id_map = None):
        self.sf_matrix = {}

        # feature_id 与 feature_idx 双向映射
        if feature_id_map is None:
            self.__feature_id_map = bidict.bidict()
        else:
            self.__feature_id_map = feature_id_map
        # category_id 与 category_idx 双向映射
        if category_id_map is None:
            self.__category_id_map = bidict.bidict()
        else:
            self.__category_id_map = category_id_map

    def clear(self):
        self.sf_matrix = {}
        self.__feature_id_map = bidict.bidict()
        self.__category_id_map = bidict.bidict()

    def get_feature_id_map(self):
        return self.__feature_id_map

    def get_category_id_map(self):
        return self.__category_id_map

    def get_num_samples(self):
        return len(self.sf_matrix)

    def get_num_features(self):
        return len(self.__feature_id_map)

    def get_num_categories(self):
        return len(self.__category_id_map)

    def get_feature_id(self, feature_idx):
        if feature_idx in (~self.__feature_id_map):
            return (~self.__feature_id_map)[feature_idx]
        else:
            return None

    def get_feature_idx(self, feature_id):
        if feature_id in self.__feature_id_map:
            return self.__feature_id_map[feature_id]
        else:
            return None

    def get_category_id(self, category_idx):
        if category_idx in (~self.__category_id_map):
            return (~self.__category_id_map)[category_idx]
        else:
            return None

    def get_category_idx(self, category_id):
        if category_id in self.__category_id_map:
            return self.__category_id_map[category_id]
        else:
            return None

    def init_cagegories(self, categories_list, sort = False):
        cl = categories_list
        if sort:
            cl = sorted(categories_list)
        for category_id in cl:
            self.__category_id_map.setdefault(category_id, len(self.__category_id_map))

    # ---------------- set_sample_category() ----------
    def set_sample_category(self, sample_id, category_id):
        category_idx = self.__category_id_map.setdefault(category_id, len(self.__category_id_map))

        if sample_id in self.sf_matrix:
            (category_old, feature_weights) = self.sf_matrix[sample_id]
            self.sf_matrix[sample_id] = (category_id, feature_weights)
        else:
            self.sf_matrix[sample_id] = (category_id, {})

        return category_idx

    # ---------------- add_sample_feature() ----------
    def add_sample_feature(self, sample_id, feature_id, feature_weight):
        feature_idx = self.__feature_id_map.setdefault(feature_id, len(self.__feature_id_map))
        if sample_id in self.sf_matrix:
            (category_id, feature_weights) = self.sf_matrix[sample_id]
            feature_weights[feature_id] = feature_weight
            self.sf_matrix[sample_id] = (category_id, feature_weights)
        else:
            self.sf_matrix[sample_id] = (category_id, {feature_id:feature_weight})


    # ---------------- get_sample_category() ----------
    def get_sample_category(self, sample_id):
        if sample_id in self.sf_matrix:
            (category_id, _) = self.sf_matrix[sample_id]
            return category_id
        else:
            return None

    # ---------------- get_sample_feature() ----------
    def get_sample_feature(self, sample_id, feature_id):
        if sample_id in self.sf_matrix:
            (category_id, feature_weights) = self.sf_matrix[sample_id]
            if feature_id in feature_weights:
                return feature_weights[feature_id]
            else:
                return None
        else:
            return None

    # ---------------- get_samples_list() ----------
    def get_samples_list(self, include_null_samples):
        samples_list = []
        for sample_id in self.sf_matrix:
            (category_id, feature_weights) = self.sf_matrix[sample_id]
            if len(feature_weights) == 0:
                if not include_null_samples:
                    continue
            samples_list.append(sample_id)
        return samples_list

    # ---------------- to_sklearn_data() ----------
    def to_sklearn_data(self, include_null_samples):
        indptr = [0]
        indices = []
        data = []

        num_samples = 0
        num_features = self.get_num_features()
        categories = []
        for sample_id in self.sf_matrix:
            (category_id, feature_weights) = self.sf_matrix[sample_id]
            if len(feature_weights) == 0:
                if not include_null_samples:
                    continue

            category_idx = self.get_category_idx(category_id)
            categories.append(category_idx)

            for feature_id in feature_weights:
                feature_idx = self.get_feature_idx(feature_id)
                indices.append(feature_idx)
                feature_weight = feature_weights[feature_id]
                data.append(feature_weight)
            indptr.append(len(indices))

            num_samples += 1

        #print data
        #print indices
        #print indptr

        if num_features != self.get_num_features():
            logging.warn(Logger.warn("SampleFeatureMatrix.to_sklearn_data() %d samples have no feature." % (self.get_num_features() - num_features)))

        X = csr_matrix((data, indices, indptr), dtype=np.float64, shape=(num_samples, num_features))
        y = categories

        return X, y

    def save(self, file_sfm):
        f = open(file_sfm, "wb+")
        feature_id_map = {}
        for feature_id in self.__feature_id_map:
            feature_id_map[feature_id] = self.__feature_id_map[feature_id]
        category_id_map = {}
        for category_id in self.__category_id_map:
            category_id_map[category_id] = self.__category_id_map[category_id]

        str_sfm = msgpack.dumps((feature_id_map, category_id_map, self.sample_categories, self.sf_matrix))

        f.write(str_sfm)
        f.close()

    def load(self, file_sfm):
        self.clear()
        f = open(file_sfm, "r")
        str_sfm = f.read()
        f.close()
        (feature_id_map, category_id_map, self.sample_categories, self.sf_matrix) = msgpack.loads(str_sfm)
        for feature_id in feature_id_map:
            self.__feature_id_map[feature_id] = feature_id_map[feature_id]
        for category_id in category_id_map:
            self.__category_id_map[category_id] = category_id_map[category_id]

    # ---------------- save_to_svm_file() ----------
    def save_to_svmfile(self, svmfile, include_null_samples):
        f = open(svmfile, 'wb+')

        for sample_id in self.sf_matrix:
            (category_id, feature_weights) = self.sf_matrix[sample_id]
            if len(feature_weights) == 0:
                if not include_null_samples:
                    continue

            category_idx = self.get_category_idx(category_id)
            f.write("%d " % (category_idx))

            features_idx = { self.get_feature_idx(feature_id):feature_weights[feature_id] for feature_id in feature_weights}
            features_list = sorted_dict(features_idx)
            for (feature_idx, feature_weight) in features_list:
                if feature_weight.__class__ is int:
                    f.write("%d:%d " % (feature_idx, feature_weight))
                else:
                    f.write("%d:%.6f " % (feature_idx, feature_weight))
            f.write("\n")

        f.close()


    # ---------------- load_from_svmfile() ----------
    #def load_from_svmfile(self, svmfile):
        #if svm_file is str:
            #f = open(svmfile, "r")
        #else:
            #f = svmfile

        #rowidx = 0
        #while True:
            #line = f.readline()
            #if line is None:
                #break
            #g = tokenize.generate_tokens(StringIO.StringIO(line).readline)
            #idx = 0
            #category = 0
            #keys = []
            #values = []
            #for tokenum, tokeval, _, _, _ in g:
                #if tokenum != 2:
                    #continue
                #if idx == 0:
                    #category = int(tokeval)
                #elif idx % 2 == 1:
                    #key = int(tokeval)
                    #keys.append(key)
                #else:
                    #value = float(tokeval)
                    #values.append(value)
                #idx += 1

            #self.set_sample_category(rowidx, category)
            #for n in range(0, len(keys)):
                #self.add_sample_feature(rowidx, keys[n], values[n])


            #if rowidx % 1000 == 0:
                #logging.debug(Logger.debug("load svm: %d" % (rowidx)))
            #rowidx += 1

        #if svmfile is str:
            #f.close()

