#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
corpus.py - 语料库（所有文本共享同一词汇表）

Samples - 样本集合。
    samples.tsm - 样本集合的词条样本矩阵。

TermSampleModel - 词条-样本模型
    词条矩阵tm_matrix，记录每一样本中所有词条分别出现的次数。
    样本矩阵sm_matrix，记录每一词条在所有出现过的样本中出现的次数。

SampleFeatureMatrix - 样本-特征矩阵。记录每一样本中所有特征的权值，及样本的类别。

CategoryFeatureMatrix - 类别-特征矩阵。记录每一类别中所有特征的权值。

'''

from __future__ import division
import sys, getopt, logging
from logger import Logger
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
from threading import Lock

from utils import *
from vocabulary import Vocabulary, SegmentMethod
from term_sample_model import TermSampleModel
from positive_degree import calculate_term_positive_degree
from feature_selection import get_terms_positive_degree_by_category
import positive_degree as pd
from classifier import Classifier
from sample_feature_matrix import SampleFeatureMatrix
from category_feature_matrix import CategoryFeatureMatrix
from protocal import decode_sample_meta
from categories import Categories
from transform import import_samples_from_xls, export_samples_to_xls, export_urls_to_xls
from feature_weighting import FeatureWeight

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

        #self.categories = Categories(self.db_content)
        #self.categories.load_categories()
        #self.categories.print_categories()

        self.tsm = TermSampleModel(self.root_dir, self.corpus.vocabulary)


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

    # ---------------- get_total_samples() ----------------
    def get_total_samples(self):
        total_samples = 0
        for i in self.db_content.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                continue
            total_samples += 1

        return total_samples


    # ---------------- get_bad_samples() ----------------
    def get_bad_samples(self):
        samples = self

        none_samples = []
        empty_samples = []
        normal_samples = []
        rowidx = 0
        for i in samples.db_content.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                continue
            (sample_id, category, date, title, key, url, msgext) = decode_sample_meta(i[1])
            (version, content, (cat1, cat2, cat3)) = msgext

            if content is None:
                none_samples.append((sample_id, url))
            elif len(content) == 0:
                empty_samples.append((sample_id, url))
            else:
                normal_samples.append((sample_id, url))

            rowidx += 1

        logging.debug(Logger.debug("Get %d bad samples. None: %d Empty: %d Normal: %d" % (len(none_samples) + len(empty_samples) +len(normal_samples), len(none_samples), len(empty_samples), len(normal_samples))))

        return none_samples, empty_samples, normal_samples

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
    def clone(self, samples_name, samples_list = None, terms_list = None):
        samples = Samples(self.corpus, samples_name)
        samples.tsm = self.tsm.clone(samples_list, terms_list)
        return samples


    # ---------------- get_samples_list() ----------------
    def get_samples_list(self):
        return os.listdir(self.samples_dir)

    # ---------------- get_categories() ----------------
    def get_categories(self):
        return self.corpus.categories

    # ---------------- import_samples() ----------------
    def import_samples(self, xls_file):
        categories = self.get_categories()

        categories.clear_categories()

        batch_content = import_samples_from_xls(self, categories, xls_file)

        categories.save_categories()

        self.db_content.Write(batch_content, sync=True)


    # ---------------- export_samples() ----------------
    def export_samples(self, xls_file):
        export_samples_to_xls(self, xls_file)


    # ---------------- export_urls() ----------------
    def export_urls(self, xls_file):
        none_samples, empty_samples, normal_samples = self.get_bad_samples()
        export_urls_to_xls(xls_file, none_samples, empty_samples, normal_samples)


    # ---------------- show() ----------------
    def show(self):
        logging.debug(Logger.debug("Do nothing in show()."))


    # ---------------- get_categories_useinfo() ----------------
    def get_categories_useinfo(self):
        categories = self.get_categories()
        db_content = self.db_content

        categories_useinfo = {}
        for category_1 in (~categories.categories_1):
            categories_useinfo[category_1] = 0
        for category_2 in (~categories.categories_2):
            categories_useinfo[category_2] = 0
        for category_3 in (~categories.categories_3):
            categories_useinfo[category_3] = 0

        unknown_categories = {}

        rowidx = 0
        for i in db_content.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                continue

            (sample_id, category_id, date, title, key, url, msgext) = decode_sample_meta(i[1])
            (version, content, (cat1, cat2, cat3)) = msgext

            if not category_id in categories_useinfo:
                if category_id in unknown_categories:
                    unknown_categories[category_id] += 1
                else:
                    unknown_categories[category_id] = 1
            else:
                categories_useinfo[category_id] += 1

            rowidx += 1

        return categories_useinfo, unknown_categories

    # ---------------- print_categories_useinfo() ----------------
    def print_categories_useinfo(self, categories_useinfo):
        categories = self.get_categories()

        categories_useinfo_list = sorted_dict(categories_useinfo)
        for (category_id, category_used) in categories_useinfo_list:
            category_name = categories.get_category_name(category_id)
            str_category = "%d - %s %d samples" % (category_id, category_name, category_used)
            print str_category

    # ---------------- query_categories() ----------------
    def query_categories(self, xls_file):
        categories = self.get_categories()

        categories_useinfo, unknown_categories = self.get_categories_useinfo()
        categories.export_categories_to_xls(categories_useinfo, xls_file)
        self.print_categories_useinfo(categories_useinfo)


    # ---------------- get_categories_1_weight_matrix() ----------------
    def get_categories_1_weight_matrix(self):
        tsm = self.tsm
        cfm = CategoryFeatureMatrix()
        sfm = SampleFeatureMatrix()

        categories = self.get_categories()
        for category_name in categories.categories_1:
            category_id = categories.categories_1[category_name]
            positive_samples_list, unlabeled_samples_list = tsm.get_samples_list_by_category_1(category_id)

            print "\n%s(%d) Positive Samples: %d Unlabeled Samples: %d" % (category_name, category_id, len(positive_samples_list), len(unlabeled_samples_list))

            terms_positive_degree = get_terms_positive_degree_by_category(tsm, positive_samples_list, unlabeled_samples_list)
            features = {}
            for term_id in terms_positive_degree:
                (pd_word, speciality, popularity) = terms_positive_degree[term_id]
                features[term_id] = pd_word
            cfm.set_features(category_id, features)

            for sample_id in positive_samples_list:
                (sample_category, sample_terms, term_map) = tsm.get_sample_row(sample_id)
                category_1_id = Categories.get_category_1_id(sample_category)
                sfm.set_sample_category(sample_id, category_1_id)
                for term_id in term_map:
                    if term_id in terms_positive_degree:
                        (pd_word, speciality, popularity) = terms_positive_degree[term_id]
                        sfm.add_sample_feature(sample_id, term_id, pd_word)
                        no_terms = False

        return cfm, sfm


    # ---------------- show_category_keywords() ----------------
    # 按二分类正例度算法，获得每个分类的关键词排序列表。
    def show_category_keywords(self, result_dir):
        if not os.path.isdir(result_dir):
            try:
                os.mkdir(result_dir)
            except OSError:
                logging.error(Logger.error("mkdir %s failed." % (result_dir)))
                return

        tsm = self.tsm

        categories = self.get_categories()
        for category_name in categories.categories_2:
            category_id = categories.categories_2[category_name]
            positive_samples_list, unlabeled_samples_list = tsm.get_samples_list_by_category_2(category_id)

            print "%s(%d) Positive Samples: %d Unlabeled Samples: %d" % (category_name, category_id, len(positive_samples_list), len(unlabeled_samples_list))

            terms_positive_degree = get_terms_positive_degree_by_category(tsm, positive_samples_list, unlabeled_samples_list)

            pd.save_terms_positive_degree(terms_positive_degree, self.corpus.vocabulary, "%s/keywords_%d_%s.txt" % (result_dir, category_id, category_name))

            samples_positive = None
            samples_unlabeled = None


    # ---------------- show_keywords_matrix() ----------------
    def show_keywords_matrix(self):
        categories = self.get_categories()

        # 计算每个词条在各个类别中使用的总次数
        # {term_id: (term_used, standard_deviation, category_info)}
        # category_info - {category_id:(term_weight, term_used_in_category, term_ratio)}
        term_category_matrix = {}

        tsm = self.tsm

        sfm_tfidf = FeatureWeight.transform(tsm, FeatureWeight.TFIDF)

        for (term_id, term_info) in tsm.term_matrix_iterator():
            (_, (term_used, term_samples, sample_map)) = term_info
            if term_used < 50:
                continue
            category_info = {}
            if term_id in term_category_matrix:
                (_, _, category_info) = term_category_matrix[term_id]
            for sample_id in sample_map:
                term_used_in_sample = sample_map[sample_id]
                (category_id, sample_terms, term_map) = tsm.get_sample_row(sample_id)

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
            ratio_mean = ratio_sum / len(categories.categories_2)

            # 计算标准差

            sum_0 = 0.0
            for category_id in category_info:
                (term_weight, term_used_in_category, term_ratio) = category_info[category_id]
                x = term_ratio - ratio_mean
                sum_0 += x * x
            #standard_deviation = math.sqrt(sum_0 / len(category_info))
            standard_deviation = math.sqrt(sum_0 / len(categories.categories_2))
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
                category_name = categories.get_category_name(category_id)

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
            #for term_id in term_map:
            terms_list = sorted_dict_by_values(term_map, reverse=True)
            for (term_id, term_used_in_sample) in terms_list:
                if term_used_in_sample <= 1:
                    continue
                term_text = self.corpus.vocabulary.get_term_text(term_id)
                #sample_terms = term_map[term_id]
                print "%s(%d): %d" % (term_text, term_id, term_used_in_sample)

        except KeyError:
            print "Sample %d not found in db_content." % (sample_id)

        db_sm = self.tsm.open_db_sm()
        try:
            str_sample_info = db_sm.Get(str(sample_id))
            (category, sample_terms, term_map) = msgpack.loads(str_sample_info)
            print ""
            print "---------------- keywords ----------------"
            print ""
            terms_list = sorted_dict_by_values(term_map, reverse = True)
            for (term_id, term_used_in_sample) in terms_list:
                if term_used_in_sample <= 1:
                    continue
                term_text = self.corpus.vocabulary.get_term_text(term_id)
                print "%s\t%d\t(id:%d)" % (term_text, term_used_in_sample, term_id)

        except KeyError:
            print "Sample %d not found in db_sm." % (sample_id)

        finally:
            self.tsm.close_db(db_sm)


    # ---------------- rebuild() ----------------
    def rebuild(self):
        self.tsm.rebuild(self.db_content)
        self.rebuild_categories()


    # ---------------- rebuild_categories() ----------------
    def rebuild_categories(self):

        samples = self
        categories = samples.get_categories()

        db_content = samples.db_content

        #categories.clear_categories()

        batch_content = leveldb.WriteBatch()
        rowidx = 0
        for i in db_content.RangeIter():
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

            category_id = categories.create_or_get_category_id(cat1, cat2, cat3)

            sample_data = (sample_id, category_id, date, title, key, url, msgext)
            rowstr = msgpack.dumps(sample_data)
            batch_content.Put(str(sample_id), rowstr)

            #if category_id != category:
            #print category_id, category, cat1, cat2, cat3
            self.tsm.set_sample_category(sample_id, category_id)
            #logging.debug(Logger.debug("[%d] %d %d=<%s:%s:%s:>" % (rowidx, sample_id, category_id, cat1, cat2, cat3)))

            rowidx += 1

        db_content.Write(batch_content, sync=True)

        self.tsm.save_sample_matrix(self.tsm.sm_matrix)

        categories.save_categories()
        categories.print_categories()


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
                logging.warn(Logger.warn("term_map %d is None." % (rowidx)))
                continue
            tm_tfidf.matrix.append((sample_id, sample_info))

            if rowidx % 1000 == 0:
                logging.debug(Logger.debug("load_tfidf_matrix() %d" % (rowidx)))

            rowidx += 1

        self.close_db(db_tfidf)

        return tm_tfidf


# ================ class Corpus ================
class Corpus():

    # ---------------- __init__() ----------------
    def __init__(self, corpus_dir):
        self.lock_meta = Lock()
        self.open(corpus_dir)

    # ---------------- __del__() ----------------
    def __del__(self):
        self.close()


    # ---------------- open_db_meta() ----------------
    def open_db_meta(self):
        logging.debug(Logger.debug("Corpus open_db_meta() %s" % (self.meta_dir) ))
        db_meta = leveldb.LevelDB(self.meta_dir)
        return db_meta

    # ---------------- close_db_meta() ----------------
    def close_db_meta(self, db_meta):
        db_meta = None


    def lock(self):
        self.lock_meta.acquire()

    def unlock(self):
        self.lock_meta.release()

    # ---------------- open() ----------------
    def open(self, corpus_dir):
        self.root_dir = corpus_dir
        if not path.isdir(corpus_dir):
            os.mkdir(corpus_dir)

        self.meta_dir = self.root_dir + "/meta"

        self.samples_dir = self.root_dir + "/samples"
        if not path.isdir(self.samples_dir):
            os.mkdir(self.samples_dir)

        self.vocabulary_dir = self.root_dir + "/vocabulary"
        self.vocabulary = Vocabulary(self.vocabulary_dir)

        self.categories_dir = self.root_dir + "/categories"
        self.categories = Categories(self.categories_dir)
        self.categories.load_categories()
        self.categories.print_categories()

    # ---------------- close() ----------------
    def close(self):
        pass


    # ---------------- acquire_sample_id() ----------------
    # 线程安全方式获取num_samples个sample_id(全Corpus唯一)。
    def acquire_sample_id(self, num_samples):
        self.lock()
        sample_id = self.get_sample_maxid()
        sample_maxid = sample_id + num_samples
        self.set_sample_maxid(sample_maxid)
        self.unlock()

        return sample_id

    def get_sample_maxid(self):
        sample_maxid = 0
        db_meta = self.open_db_meta()
        try:
            str_maxid = db_meta.Get("__sample_maxid__")
            sample_maxid = int(str_maxid)
        except KeyError:
            db_meta.Put("__sample_maxid__", "0")
        self.close_db_meta(db_meta)

        return sample_maxid

    def set_sample_maxid(self, sample_maxid):
        db_meta = self.open_db_meta()
        db_meta.Put("__sample_maxid__", str(sample_maxid))
        self.close_db_meta(db_meta)

    # ---------------- export_svm_file() ----------------
    def export_svm_file(self, samples_name, svm_file):
        samples = Samples(self, samples_name)

        logging.debug(Logger.debug("Export svm file..."))
        tm_tfidf = samples.load_tfidf_matrix()

        save_term_matrix_as_svm_file(tm_tfidf, svm_file)


    # ---------------- transform_sensitive_terms() ----------------
    def transform_sensitive_terms(self, sensitive_words, vocabulary):
        sensitive_terms = {}
        if not sensitive_words is None:
            for word in sensitive_words:
                w = sensitive_words[word]
                term_id = vocabulary.get_term_id(word)
                sensitive_terms[term_id] = w
        return sensitive_terms

    # ---------------- query_by_id() ----------------
    def query_by_id(self, samples_positive, samples_unlabeled, sample_id):
        tsm_positive = samples_positive.tsm
        tsm_unlabeled = samples_unlabeled.tsm

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
            #(_, category, date, title, key, url, content) = msgpack.loads(sample_content)

            (_, category, date, title, key, url, msgext) = decode_sample_meta(sample_content)
            (version, content, (cat1, cat2, cat3)) = msgext

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
            #for term_id in term_map:
            terms_list = sorted_dict_by_values(term_map, reverse=True)
            for (term_id, term_used_in_sample) in terms_list:
                term_text = self.vocabulary.get_term_text(term_id)
                #term_used_in_sample = term_map[term_id]
                print "%s(%d): %d" % (term_text, term_id, term_used_in_sample)


        except KeyError:
            print "Sample %d not found in db_content." % (sample_id)

        db_sm = samples_unlabeled.tsm.open_db_sm()
        try:
            str_sample_info = db_sm.Get(str(sample_id))
            (category, sample_terms, term_map) = msgpack.loads(str_sample_info)
            print ""
            print "---------------- keywords ----------------"
            print ""
            terms = {}
            for term_id in term_map:
                term_text = self.vocabulary.get_term_text(term_id)
                term_used = term_map[term_id]
                (pd_word, speciality, popularity) = calculate_term_positive_degree(term_id, tsm_positive, tsm_unlabeled, sensitive_terms)
                terms[term_id] = (pd_word, speciality, popularity, term_used, term_text)

            terms_list = sorted_dict_by_values(terms, reverse = True)
            for (term_id, (pd_word, speciality, popularity, term_used, term_text)) in terms_list:
                print "%s\t%d\t[%.6f,%.6f,%.6f]\t(id:%d)" % (term_text, term_used, pd_word, speciality, popularity, term_id)

        except KeyError:
            print "Sample %d not found in db_sm." % (sample_id)

        samples_unlabeled.tsm.close_db(db_sm)


