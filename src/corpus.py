#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
corpus.py - 语料库（所有文本共享同一词汇表）

Samples - 样本集合。
    samples.tsm - 样本集合的词条样本矩阵。

TermSampleMatrix - 词条样本矩阵
    词条矩阵tm_matrix，记录每一样本中所有词条分别出现的次数。
    样本矩阵sm_matrix，记录每一词条在所有出现过的样本中出现的次数。

SampleFeatureMatrix - 样本-特征矩阵。记录每一样本中所有特征的权值，及样本的类别。

CategoryFeatureMatrix - 类别-特征矩阵。记录每一类别中所有特征的权值。

'''

from __future__ import division
import sys, getopt, logging
import os
from os import path
import math
import json, leveldb, msgpack
import bidict
import requests
from datetime import datetime
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
import numpy as np
from scipy.sparse import csr_matrix

from utils import *
from vocabulary import Vocabulary, SegmentMethod
from term_sample_matrix import TermSampleMatrix
from feature_selection import calculate_term_positive_degree, get_terms_positive_degree_by_category
import positive_degree as pd
from classifier import Classifier
from sample_feature_matrix import SampleFeatureMatrix
from category_feature_matrix import CategoryFeatureMatrix
from protocal import decode_sample_meta
from categories import Categories
from transform import import_samples_from_xls, export_samples_to_xls, export_urls_to_xls

# ================ class Samples ================
class Samples():

    # ---------------- __init__() ----------------
    def __init__(self, corpus, name):
        self.corpus = corpus
        self.name = name

        self.N = 4
        self.MAX_FETS = 3000

        self.root_dir = corpus.samples_dir + "/" + self.name
        if not path.isdir(self.root_dir):
            os.mkdir(self.root_dir)

        self.tfidf_dir = self.root_dir + "/tfidf"
        #self.db_tfidf = leveldb.LevelDB(self.tfidf_dir)

        self.meta_dir = self.root_dir + "/meta"
        #self.db_meta = leveldb.LevelDB(self.meta_dir)

        self.content_dir = self.root_dir + "/content"
        self.db_content = leveldb.LevelDB(self.content_dir)

        self.sample_maxid = self.get_sample_maxid()

        self.categories = Categories(self.db_content)
        self.categories.load_categories()
        self.categories.print_categories()

        self.tsm = TermSampleMatrix(self.root_dir, self.corpus.vocabulary)

    def get_term_matrix(self):
        return self.tsm.tm_matrix

    def get_sample_matrix(self):
        return self.tsm.sm_matrix

    def merge(self, other_samples):
        self.tsm.merge(other_samples.tsm)

    # ---------------- clear() ----------------
    def clear(self):
        pass

    def close_db(self, db):
        db = None

    # ---------------- get_int_value_in_db() ----------------
    def get_int_value_in_db(self, db, key):
        try:
            str_value = db.Get(key)
            return int(str_value)
        except KeyError:
            return 0


    # ---------------- set_int_value_in_db() ----------------
    def set_int_value_in_db(self, db, key, value):
        db.Put(key, str(value))


    # ---------------- get_maxid_in_db() ----------------
    def get_maxid_in_db(self, db):
        try:
            maxid = self.get_int_value_in_db(db, "__maxid__")
            return maxid
        except KeyError:
            return 0

    # ---------------- set_maxid_in_db() ----------------
    def set_maxid_in_db(self, db, maxid):
        self.set_int_value_in_db(db, "__maxid__", maxid)

    # ---------------- get_sample_maxid() ----------------
    def get_sample_maxid(self):
        return self.get_maxid_in_db(self.db_content)

    # ---------------- set_sample_maxid() ----------------
    def set_sample_maxid(self, maxid):
        self.set_maxid_in_db(self.db_content, maxid)
        self.sample_maxid = maxid

    # ---------------- get_total_samples() ----------------
    def get_total_samples(self):
        total_samples = 0
        for i in self.db_content.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                continue
            total_samples += 1

        return total_samples


    # ---------------- get_sample_meta() ----------------
    '''
    sample_meta:
        (sample_id, category, date, title, key, url, content)
    '''
    def get_sample_meta(self, sample_id):
        try:
            str_sample_meta = self.db_content.Get(str(sample_id))
            return decode_sample_meta(str_sample_meta)

        except KeyError:
            return None

    # ---------------- clone() ----------------
    def clone(self, samples_name, samples_list):
        samples = Samples(self.corpus, samples_name)
        samples.tsm = self.tsm.clone(samples_list)
        return samples


    # ---------------- get_samples_list() ----------------
    def get_samples_list(self):
        return os.listdir(self.samples_dir)


    # ---------------- import_samples() ----------------
    def import_samples(self, xls_file):
        self.clear_categories()

        max_sample_id, batch_content = import_samples_from_xls(self, self.categories, self.max_sample_id, xls_file)

        self.save_categories()

        self.db_content.Write(batch_content, sync=True)
        self.set_sample_maxid(max_sample_id)


    # ---------------- export_samples() ----------------
    def export_samples(self, xls_file):
        export_samples_to_xls(self, xls_file)


    # ---------------- export_urls() ----------------
    def export_urls(self, xls_file):
        none_samples, empty_samples, normal_samples = self.get_bad_samples()
        export_urls_to_xls(xls_file, none_samples, empty_samples, normal_samples)


    # ---------------- show() ----------------
    def show(self):
        logging.debug("Do nothing in show().")


    # ---------------- query_categories() ----------------
    def query_categories(self, xls_file):
        categories = self.categories.get_categories_list()
        self.categories.export_categories_to_xls(categories, xls_file)
        self.categories.print_categories_info(categories)


    # ---------------- get_categories_1_weight_matrix() ----------------
    def get_categories_1_weight_matrix(self):
        tsm = self.tsm
        tm_matrix = tsm.tm_matrix
        sm_matrix = tsm.sm_matrix
        cfm = CategoryFeatureMatrix()
        sfm = SampleFeatureMatrix()
        print "len of tm_matrix: %d" % (len(tm_matrix))

        for category_name in self.categories.categories_1:
            category_id = self.categories.categories_1[category_name]
            positive_samples_list, unlabeled_samples_list = tsm.divide_samples_by_category_1(category_id, True)

            print "%s(%d) Positive Samples: %d Unlabeled Samples: %d" % (category_name, category_id, len(positive_samples_list), len(unlabeled_samples_list))

            terms_positive_degree = get_terms_positive_degree_by_category(tsm, positive_samples_list, unlabeled_samples_list)
            features = {}
            for term_id in terms_positive_degree:
                (pd_word, specialty, popularity) = terms_positive_degree[term_id]
                features[term_id] = pd_word
            cfm.set_features(category_id, features)

            for sample_id in positive_samples_list:
                (sample_category, sample_terms, term_map) = tm_matrix[sample_id]
                category_id_1 = self.categories.get_category_id_1(sample_category)
                sfm.set_sample_category(sample_id, category_id_1)
                for term_id in term_map:
                    if term_id in terms_positive_degree:
                        (pd_word, specialty, popularity) = terms_positive_degree[term_id]
                        sfm.add_sample_feature(sample_id, term_id, pd_word)

        return cfm, sfm


    # ---------------- show_category_keywords() ----------------
    # 按二分类正例度算法，获得每个分类的关键词排序列表。
    def show_category_keywords(self):
        tsm = self.tsm
        tm_matrix = tsm.tm_matrix
        sm_matrix = tsm.sm_matrix

        print "len of tm_matrix: %d" % (len(tm_matrix))

        for category_name in self.categories.categories_2:
            category_id = self.categories.categories_2[category_name]
            positive_samples_list, unlabeled_samples_list = tsm.divide_samples_by_category_2(category_id, True)

            print "%s(%d) Positive Samples: %d Unlabeled Samples: %d" % (category_name, category_id, len(positive_samples_list), len(unlabeled_samples_list))

            terms_positive_degree = get_terms_positive_degree_by_category(tsm, positive_samples_list, unlabeled_samples_list)

            pd.save_terms_positive_degree(terms_positive_degree, self.corpus.vocabulary, "./result/keywords_%d_%s.txt" % (category_id, category_name))

            samples_positive = None
            samples_unlabeled = None


    # ---------------- show_keywords_matrix() ----------------
    def show_keywords_matrix(self):
        # 计算每个词条在各个类别中使用的总次数
        # {term_id: (term_used, standard_deviation, category_info)}
        # category_info - {category_id:(term_weight, term_used_in_category, term_ratio)}
        term_category_matrix = {}

        tm_matrix = self.tsm.tm_matrix
        sm_matrix = self.tsm.sm_matrix

        sfm_tfidf = self.tsm.tranform_tfidf()

        for term_id in sm_matrix:
            (_, (term_used, term_samples, sample_map)) = sm_matrix[term_id]
            if term_used < 50:
                continue
            category_info = {}
            if term_id in term_category_matrix:
                (_, _, category_info) = term_category_matrix[term_id]
            for sample_id in sample_map:
                term_used_in_sample = sample_map[sample_id]
                (category_id, sample_terms, term_map) = tm_matrix[sample_id]

                term_weight = 0.0
                term_used_in_category = 0
                term_ratio = 0.0
                term_ratio_variance = 0.0
                if category_id in category_info:
                    (term_weight, term_used_in_category, term_ratio) = category_info[category_id]

                v = sfm_tfidf.get_sample_feature(sample_id, term_id)
                if v is None:
                    continue

                category_info[category_id] = (term_weight + v, term_used_in_category + term_used_in_sample,  term_ratio)

                term_category_matrix[term_id] = (term_used, 0.0, category_info)

        # 计算每个词条在各个类别中的使用占比。
        for term_id in term_category_matrix:
            (term_used, _, category_info) = term_category_matrix[term_id]
            # 计算词条使用占比
            term_weight_sum = 0.0
            for category_id in category_info:
                (term_weight, term_used_in_category, _) = category_info[category_id]
                term_weight_sum += term_weight
                #term_weight_sum += term_used_in_category

            ratio_sum = 0.0
            for category_id in category_info:
                (term_weight, term_used_in_category, _) = category_info[category_id]
                term_ratio = term_weight / term_weight_sum
                category_info[category_id] = (term_weight, term_used_in_category, term_ratio)
                ratio_sum += term_ratio

            term_category_matrix[term_id] = (term_used, 0.0, category_info)

            #ratio_mean = ratio_sum / len(category_info)
            ratio_mean = ratio_sum / len(self.categories.categories_2)

            # 计算标准差

            sum_0 = 0.0
            for category_id in category_info:
                (term_weight, term_used_in_category, term_ratio) = category_info[category_id]
                x = term_ratio - ratio_mean
                sum_0 += x * x
            #standard_deviation = math.sqrt(sum_0 / len(category_info))
            standard_deviation = math.sqrt(sum_0 / len(self.categories.categories_2))
            term_category_matrix[term_id] = (term_used, standard_deviation, category_info)

        # 输出结果
        # 按标准差从大到小排序
        terms_by_sd = {}
        for term_id in term_category_matrix:
            (term_used, standard_deviation, category_info) = term_category_matrix[term_id]
            terms_by_sd[term_id] = standard_deviation

        rowidx = 0
        terms_by_sd_list = sorted_dict_by_values(terms_by_sd, reverse = True)
        for (term_id, standard_deviation) in terms_by_sd_list:
            (term_used, _, category_info) = term_category_matrix[term_id]
            term_text = self.corpus.vocabulary.get_term_text(term_id)

            str_term_categories = u""
            category_info_list = sorted_dict_by_values(category_info, reverse = True)
            for (category_id, (term_weight, term_used_in_category, term_ratio)) in category_info_list:
                category_name = self.categories.get_category_name(category_id)

                str_term_categories += " <%s[%d]: %.2f%% (%d)> " % (category_name, category_id, term_ratio * 100, term_used_in_category)

            print "--------------------------------"
            print "<%d/%d> %s(%d) sd:%.6f %d used. %s" % (rowidx, len(terms_by_sd_list), term_text, term_id, standard_deviation, term_used, str_term_categories)
            rowidx += 1


    # ---------------- query_by_id() ----------------
    def query_by_id(self, sample_id):
        try:
            sample_content = self.db_content.Get(str(sample_id))
            (_, category, date, title, key, url, msgext) = decode_sample_meta(sample_content)
            (version, content, (cat1, cat2, cat3)) = msgext

            print "sample id: %d" % (sample_id)
            print "category: %d" % (category)
            print "key: %s" % (key)
            print "url: %s" % (url)
            print "date: %s" % (date)
            print "title: %s" % (title)
            print "---------------- content ----------------"
            print "%s" % (content)
            sample_terms, term_map = self.corpus.vocabulary.seg_content(content)
            print "sample_terms: %d terms_count: %d" % (sample_terms, len(term_map))
            for term_id in term_map:
                term_text = self.corpus.vocabulary.get_term_text(term_id)
                sample_terms = term_map[term_id]
                print "%s(%d): %d" % (term_text, term_id, sample_terms)

        except KeyError:
            print "Sample %d not found in db_content." % (sample_id)

        db_tm = self.tsm.open_db_tm()
        try:
            str_sample_info = db_tm.Get(str(sample_id))
            (category, sample_terms, term_map) = msgpack.loads(str_sample_info)
            print ""
            print "---------------- keywords ----------------"
            print ""
            terms_list = sorted_dict_by_values(term_map, reverse = True)
            for (term_id, term_used_in_sample) in terms_list:
                term_text = self.corpus.vocabulary.get_term_text(term_id)
                print "%s\t%d\t(id:%d)" % (term_text, term_used_in_sample, term_id)

        except KeyError:
            print "Sample %d not found in db_tm." % (sample_id)

        finally:
            self.tsm.close_db(db_tm)


    # ---------------- rebuild() ----------------
    def rebuild(self):
        self.tsm.rebuild(self.db_content)

    # ---------------- load() ----------------
    def load(self):
        self.tsm.load()

    # ---------------- save_tfidf_matrix() -----------------
    def save_tfidf_matrix(self, tm_tfidf):
        db_tfidf = self.open_db_tfidf()


        for sample_id in tm_tfidf:
            sample_info = tm_tfidf[sample_id]
            db_tfidf.Put(str(sample_id), msgpack.dumps(sample_info))

        self.close_db(db_tfidf)


    # ---------------- load_tfidf_matrix() ----------------
    def load_tfidf_matrix(self):
        db_tfidf = self.open_db_tfidf()

        tm_tfidf = TermMatrix()
        rowidx = 0
        for i in db_tfidf.RangeIter():
            row_id = i[0]
            if row_id[0:2] == "__":
                continue
            sample_id = int(row_id)
            sample_info = msgpack.loads(i[1])
            if term_map is None:
                logging.warning("term_map %d is None." % (rowidx))
                continue
            tm_tfidf.matrix.append((sample_id, sample_info))

            if rowidx % 1000 == 0:
                logging.debug("load_tfidf_matrix() %d" % (rowidx))

            rowidx += 1

        self.close_db(db_tfidf)

        return tm_tfidf


# ================ class Corpus ================
class Corpus():

    # ---------------- __init__() ----------------
    def __init__(self, corpus_dir):
        self.open(corpus_dir)

    # ---------------- __del__() ----------------
    def __del__(self):
        self.close()


    # ---------------- open() ----------------
    def open(self, corpus_dir):
        self.root_dir = corpus_dir
        if not path.isdir(corpus_dir):
            os.mkdir(corpus_dir)
        self.samples_dir = self.root_dir + "/samples"
        if not path.isdir(self.samples_dir):
            os.mkdir(self.samples_dir)

        self.vocabulary_dir = self.root_dir + "/vocabulary"
        self.vocabulary = Vocabulary(self.vocabulary_dir)


    # ---------------- close() ----------------
    def close(self):
        pass


    # ---------------- export_svm_file() ----------------
    def export_svm_file(self, samples_name, svm_file):
        samples = Samples(self, samples_name)

        logging.debug("Export svm file...")
        tm_tfidf = samples.load_tfidf_matrix()

        save_term_matrix_as_svm_file(tm_tfidf, svm_file)


    def transform_sensitive_terms(self, sensitive_words, vocabulary):
        sensitive_terms = {}
        if not sensitive_words is None:
            for word in sensitive_words:
                w = sensitive_words[word]
                term_id = vocabulary.get_term_id(word)
                sensitive_terms[term_id] = w
        return sensitive_terms

    def query_by_id(self, samples_positive, samples_unlabeled, sample_id):

        sensitive_words = {
                ##u"立案":3.0,
                ##u"获刑":3.0,
                ##u"受贿":3.0,
                ##u"有期徒刑":3.0,
                ##u"宣判":3.0,
                ##u"审计":2.0,
                ##u"调查":2.0
                }

        sensitive_terms = self.transform_sensitive_terms(sensitive_words, self.vocabulary)

        try:
            sample_content = samples_unlabeled.db_content.Get(str(sample_id))
            (_, category, date, title, key, url, content) = msgpack.loads(sample_content)
            print "sample id: %d" % (sample_id)
            print "category: %d" % (category)
            print "key: %s" % (key)
            print "url: %s" % (url)
            print "date: %s" % (date)
            print "title: %s" % (title)
            print "---------------- content ----------------"
            #print "%s" % (content)

            sample_terms, term_map = self.vocabulary.seg_content(content)
            print "sample_terms: %d terms_count: %d" % (sample_terms, len(term_map))
            for term_id in term_map:
                term_text = self.vocabulary.get_term_text(term_id)
                sample_terms = term_map[term_id]
                print "%s(%d): %d" % (term_text, term_id, sample_terms)


        except KeyError:
            print "Sample %d not found in db_content." % (sample_id)

        db_tm = samples_unlabeled.tsm.open_db_tm()
        try:
            str_sample_info = db_tm.Get(str(sample_id))
            (category, sample_terms, term_map) = msgpack.loads(str_sample_info)
            print ""
            print "---------------- keywords ----------------"
            print ""
            terms = {}
            for term_id in term_map:
                term_text = self.vocabulary.get_term_text(term_id)
                term_used = term_map[term_id]
                (pd_word, specialty, popularity) = calculate_term_positive_degree(term_id, samples_positive, samples_unlabeled, self.sensitive_terms)
                terms[term_id] = (pd_word, specialty, popularity, term_used, term_text)

            terms_list = sorted_dict_by_values(terms, reverse = True)
            for (term_id, (pd_word, specialty, popularity, term_used, term_text)) in terms_list:
                print "%s\t%d\t[%.6f,%.6f,%.6f]\t(id:%d)" % (term_text, term_used, pd_word, specialty, popularity, term_id)

        except KeyError:
            print "Sample %d not found in db_tm." % (sample_id)

        samples_unlabeled.tsm.close_db(db_tm)


