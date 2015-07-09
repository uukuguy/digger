#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
term_sample_matrix.py 词条-样本矩阵

    SampleMatrix: 横着 样本矩阵，描述每个样本中词条的分布情况。以样本为序(sample_id)，包括每个样本的元数据信息
        [{sample_id: (category, sample_terms, term_map)}]

        term_map: 样本中词条分布情况。所有词条(term_id)在样本中的使用次数(term_used_in_sample)。
        {term_id:term_used_in_sample}

    TermMatrix:词条矩阵，描述每个词条的样本分布情况。以词条为序(term_id)，包括每个词条的元数据信息。
        [{term_id: (term_used, term_samples, sample_map)}]
            term_used - term在样本集合中出现的总次数。
            term_samples - term在样本集合中出现过的样本总数。
        sample_map: 词条的样本分布情况。所有使用该词条的样本集合。
        {sample_id:term_used_in_sample}

'''

from __future__ import division
import logging
import os
from os import path
import math
import shutil
import msgpack
import leveldb
import random
from utils import is_chinese_word

from scipy.sparse import csr_matrix
import numpy as np

from utils import save_as_svm_file
from vocabulary import Vocabulary
from sample_feature_matrix import SampleFeatureMatrix
from categories import Categories

class TermSampleMatrixIterator():
    def __init__(self, itemiteror):
        self.itemiteror = itemiteror


    def next(self):
        (key, value) = self.itemiteror.next()
        if key.__class__ is str and key.startswith('__'):
            (key, value) = self.itemiteror.next()
        return (key, value)


class TermSampleMatrix():

    # ---------------- __init__() ----------
    def __init__(self, root_dir, vocabulary):
        self.root_dir = root_dir
        self.vocabulary = vocabulary

        self.sm_matrix = {}
        self.sm_dir = self.root_dir + "/sm"

        self.total_terms_used = 0
        self.tm_matrix = {}
        self.tm_dir = self.root_dir + "/tm"

        self.categories = []
        self.sample_max_id = 0

    # ---------------- sample_matrix() ----------------
    def sample_matrix(self):
        return self.sm_matrix
        #return TermSampleMatrixIterator(self.sm_matrix.iteritems())

    # ---------------- get_total_samples() ----------------
    def get_total_samples(self):
        return len(self.sm_matrix)

    # ---------------- get_sample_row() ----------------
    def get_sample_row(self, sample_id):
        return self.sm_matrix[sample_id]

    # ---------------- term_matrix() ----------------
    def term_matrix(self):
        return self.tm_matrix
        #return TermSampleMatrixIterator(self.tm_matrix.iteritems())

    # ---------------- get_total_terms() ----------------
    def get_total_terms(self):
        return len(self.tm_matrix)

    # ---------------- get_term_row() ----------------
    def get_term_row(self, term_id):
        return self.tm_matrix[term_id]


    # ---------------- open_db_tm() ----------------
    def open_db_tm(self):
        logging.debug("open_db_tm() %s" % (self.tm_dir) )
        db_tm = leveldb.LevelDB(self.tm_dir)
        return db_tm

    # ---------------- open_db_sm() ----------------
    def open_db_sm(self):
        logging.debug("open_db_sm() %s" % (self.sm_dir) )
        db_sm = leveldb.LevelDB(self.sm_dir)
        return db_sm

    # ---------------- close_db() ----------------
    def close_db(self, db):
        db = None


    # ---------------- clone() ----------------
    def clone(self, samples_list = None):
        term_sample_matrix = TermSampleMatrix("", self.vocabulary)
        if samples_list is None:
            for sample_id in self.sm_matrix:
                sample_info = self.sm_matrix[sample_id]
                term_sample_matrix.sm_matrix[sample_id] = sample_info
            term_sample_matrix.sample_max_id = self.sample_max_id
        else:
            term_sample_matrix.sample_max_id = 0
            for sample_id in samples_list:
                sample_info = self.sm_matrix[sample_id]
                term_sample_matrix.sm_matrix[sample_id] = sample_info
                if sample_id >= term_sample_matrix.sample_max_id:
                    term_sample_matrix.sample_max_id = sample_id + 1

        term_sample_matrix.rebuild_term_matrix()

        return term_sample_matrix

    # ---------------- rebuild_categories() ----------------
    def rebuild_categories(self):
        self.categories = []
        for sample_id in self.sm_matrix:
            (category_id, _, _) = self.sm_matrix[sample_id]
            if not category_id in self.categories:
                self.categories.append(category_id)
        return self.categories

    # ---------------- get_categories() ----------------
    def get_categories(self):
        return self.categories

    # ---------------- set_sample_category() ----------------
    def set_sample_category(self, sample_id, category_id):
        if sample_id in self.sm_matrix:
            (category_old, sample_terms, term_map) = self.sm_matrix[sample_id]
            self.sm_matrix[sample_id] = (category_id, sample_terms, term_map)
            #logging.debug("set_sample_category(%d, %d) old_category:%d" % (sample_id, category_id, category_old))

    # ---------------- get_sample_category() ----------------
    def get_sample_category(self, sample_id):
        if sample_id in self.sm_matrix:
            (category_id, _, _) = self.sm_matrix[sample_id]
            return category_id
        else:
            return None

    # ---------------- merge() ----------------
    def merge(self, other_tsm, renewid = False):
        logging.debug("Merge %d samples into %d samples." % (other_tsm.get_total_samples(), self.get_total_samples()))

        sample_max_id = self.sample_max_id
        rowidx = 0
        for sample_id in other_tsm.sm_matrix:
            sample_info = other_tsm.sm_matrix[sample_id]

            if renewid:
                new_sample_id = sample_id + sample_max_id
            else:
                new_sample_id = sample_id
            self.sm_matrix[new_sample_id] = sample_info
            #if rowidx % 1000 == 0:
                #logging.debug("Merge sample matrix %d/%d" % (rowidx, len(other_tsm.sm_matrix)))
            rowidx += 1

        logging.debug("Merge %d terms into %d terms." % (other_tsm.get_total_terms(), self.get_total_terms()))
        rowidx = 0
        for term_id in other_tsm.tm_matrix:
            (_, (term_used, term_samples, sample_map)) = other_tsm.tm_matrix[term_id]
            new_sample_map = {}
            for sample_id in sample_map:
                term_used_in_sample = sample_map[sample_id]
                if renewid:
                    new_sample_id = sample_id + sample_max_id
                else:
                    new_sample_id = sample_id
                new_sample_map[new_sample_id] = term_used_in_sample

            term_used_0 = 0
            term_samples_0 = 0
            sample_map_0 = {}
            if term_id in self.tm_matrix:
                (_, (term_used_0, term_samples_0, sample_map_0)) = self.tm_matrix[term_id]
            self.tm_matrix[term_id] = (term_id, (term_used + term_used_0, term_samples + term_samples_0, dict(sample_map_0, **new_sample_map)))

            #if rowidx % 1000 == 0:
                #logging.debug("Merge term matrix %d/%d" % (rowidx, len(other_tsm.tm_matrix)))
            rowidx += 1

        self.rebuild_categories()


    # ---------------- save_sample_matrix() ----------
    def save_sample_matrix(self, sm_matrix):
        batch_sm = leveldb.WriteBatch()
        rowidx = 0
        for sample_id in sm_matrix:
            sample_info = sm_matrix[sample_id]
            #print sample_info

            batch_sm.Put(str(sample_id), msgpack.dumps(sample_info))

            if rowidx % 1000 == 0:
                logging.debug("save_sample_matrix() %d %d" % (rowidx, sample_id))
                print sample_info

            rowidx += 1

        if path.isdir(self.tm_dir):
            shutil.rmtree(self.tm_dir)
        db_sm = self.open_db_sm()
        db_sm.Write(batch_sm, sync=True)
        self.close_db(db_sm)


    # ---------------- save_term_matrix() ----------
    def save_term_matrix(self, tm_matrix):
        batch_tm = leveldb.WriteBatch()
        rowidx = 0
        for term_id in tm_matrix:
            #print tm_matrix[term_id]
            (_, term_info) = tm_matrix[term_id]
            batch_tm.Put(str(term_id), msgpack.dumps(term_info))

            if rowidx % 1000 == 0:
                logging.debug("save_term_matrix() %d %d" % (rowidx, term_id))
            rowidx += 1

        if path.isdir(self.tm_dir):
            shutil.rmtree(self.tm_dir)
        db_tm = self.open_db_tm()
        db_tm.Write(batch_tm, sync=True)
        db_tm.Put("__total_terms_used__", str(self.total_terms_used))
        self.close_db(db_tm)


    # ---------------- save() ----------
    def save(self):
        logging.debug("Save vocabulary ...")
        self.vocabulary.save()

        logging.debug("Save sample matrix ...")
        self.save_sample_matrix(self.sm_matrix)

        logging.debug("Save term matrix ...")
        self.save_term_matrix(self.tm_matrix)


    # ---------------- load_sample_matrix() ----------
    def load_sample_matrix(self, db_sm):
        sm_matrix = {}
        rowidx = 0
        for i in db_sm.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                continue

            sample_id = int(row_id)
            sample_info = msgpack.loads(i[1])
            #print sample_info
            sm_matrix[sample_id] = sample_info
            #print sample_info

            if rowidx % 1000 == 0:
                logging.debug("load_sample_matrix() %d" % (rowidx))

            rowidx += 1

        return sm_matrix

    # ---------------- load_term_matrix() ----------
    def load_term_matrix(self, db_tm):
        tm_matrix = {}
        rowidx = 0
        for i in db_tm.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                if row_id == "__total_terms_used__":
                    self.total_terms_used = int(i[1])
                continue
            term_id = int(row_id)
            term_info = msgpack.loads(i[1])
            tm_matrix[term_id] = (term_id, term_info)
            #print tm_matrix[term_id]

            if rowidx % 1000 == 0:
                logging.debug("load_term_matrix() %d" % (rowidx))

            rowidx += 1

        logging.debug("load total terms: %d" % (len(tm_matrix)))
        return tm_matrix

    # ---------------- load() ----------
    def load(self):
        self.sm_matrix = {}
        db_sm = self.open_db_sm()
        self.sm_matrix = self.load_sample_matrix(db_sm)
        self.close_db(db_sm)

        self.tm_matrix = {}
        db_tm = self.open_db_tm()
        self.tm_matrix = self.load_term_matrix(db_tm)
        self.close_db(db_tm)

        self.rebuild_categories()

        return self.tm_matrix, self.sm_matrix


    # ---------------- rebuild_sample_matrix() ----------
    def rebuild_sample_matrix(self, db_content):
        sm_matrix = {}
        rowidx = 0
        for i in db_content.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                continue
            (sample_id, category, date, title, key, url, msgext) = msgpack.loads(i[1])
            version = ""
            cat1 = ""
            cat2 = ""
            cat3 = ""
            if msgext.__class__ is str:
                content = msgext
            else:
                (version, content, extdata) = msgext
                if version == "1":
                    (cat1, cat2, cat3) = extdata

            if title is None:
                title = u""
            else:
                title = title.decode('utf-8')
            if content is None:
                content = u""
            else:
                content = content.decode('utf-8')

            sample_terms, term_map = self.vocabulary.seg_content(title + content)

            if sample_terms == 1 or sample_terms == 0:
                logging.warn("!!!!!!!!!! [%d] sample_terms == %d" % (rowidx, sample_terms))

            if len(cat1) > 0:
                category = categories.create_or_get_category_id(cat1, cat2, cat3)

            sm_matrix[sample_id] = (category, sample_terms, term_map)
            if sample_id >= self.sample_max_id:
                self.sample_max_id = sample_id + 1

            if rowidx % 100 == 0:
                logging.debug("rebuild_sample_matrix() %d %d %d %s %s" % (rowidx, sample_id, category, date, title))
            rowidx += 1

        return sm_matrix


    # ---------------- rebuild_term_matrix() ----------
    def rebuild_term_matrix(self):
        sm_matrix = self.sm_matrix
        total_terms_used = 0
        tm_matrix = {}
        rowidx = 0
        for sample_id in sm_matrix:
            (category, sample_terms, term_map) = sm_matrix[sample_id]
            for term_id in term_map:
                term_used = 0
                term_samples = 0
                sample_map = {}
                if term_id in tm_matrix:
                    (_, (term_used, term_samples, sample_map)) = tm_matrix[term_id]

                term_used_in_sample = term_map[term_id]
                #term_used_in_sample = 1

                total_terms_used += term_used_in_sample

                term_used += term_used_in_sample
                sample_map[sample_id] = term_used_in_sample

                tm_matrix[term_id] = (term_id, (term_used, term_samples + 1, sample_map))

            #if rowidx % 1000 == 0:
                #logging.debug("rebuild_sample_matrix() %d/%d" % (rowidx, len(tm_matrix)))
            rowidx += 1

        self.total_terms_used = total_terms_used
        self.tm_matrix = tm_matrix

        return total_terms_used, tm_matrix


    # ---------------- rebuild() ----------
    def rebuild(self, db_content, do_save = True):

        logging.debug("Rebuild sample matrix...")
        self.sm_matrix = self.rebuild_sample_matrix(db_content)

        if do_save:
            logging.debug("Save vocabulary ...")
            self.vocabulary.save()

        logging.debug("Save sample matrix ...")
        self.save_sample_matrix(self.sm_matrix)

        logging.debug("Calculate term matrix...")
        self.rebuild_term_matrix()

        if do_save:
            logging.debug("Save term matrix ...")
            self.save_term_matrix(self.tm_matrix)

        logging.debug("Rebuild TermSampleMatrix Done!")

        self.rebuild_categories()

        return self.sm_matrix, self.tm_matrix


    # ---------------- to_sklearn_data() ----------
    def to_sklearn_data(self):
        indptr = [0]
        indices = []
        data = []
        categories = []
        terms = {}
        category_map = {}
        for sample_id in self.sm_matrix:
            (category_id, sample_terms, term_map) = self.sm_matrix[sample_id]

            category_1_id = Categories.get_category_1_id(category_id)
            category_id_1 = category_1_id / 1000000
            category_idx = category_map.setdefault(category_id_1, len(category_map))
            categories.append(category_idx)
            #categories.append(category)

            for term_id in term_map:
                term_idx = terms.setdefault(term_id, len(terms))
                indices.append(term_idx)
                term_used_in_sample = term_map[term_id]
                data.append(term_used_in_sample)
            indptr.append(len(indices))

        rows = len(self.sm_matrix)
        cols = len(terms)
        print rows, cols
        X = csr_matrix((np.array(data), np.array(indices), np.array(indptr)), shape = (rows, cols))
        y = categories

        return X, y, terms, category_map

    # ---------------- tranform_tfidf() ----------
    def tranform_tfidf(self):
        sm_matrix = self.sm_matrix
        tm_matrix = self.tm_matrix

        sfm = SampleFeatureMatrix()
        total_samples = len(sm_matrix)

        rowidx = 0
        for sample_id in sm_matrix:
            (category, sample_terms, term_map) = sm_matrix[sample_id]

            sfm.set_sample_category(sample_id, category)
            colidx = 0
            for term_id in term_map:
                term_used = term_map[term_id]
                tf = term_used / sample_terms
                (_, (_, term_samples, _)) = tm_matrix[term_id]
                idf = math.log(total_samples/term_samples)
                tfidf = tf * idf
                sfm.add_sample_feature(sample_id, term_id, tfidf)
                colidx += 1

            rowidx += 1

        return sfm


    # ---------------- get_samples_list() ----------------
    def get_samples_list(self, by_category_1 = None, by_category_2 = None):
        if not by_category_1 is None:
            L = get_samples_list_by_category_1(by_category_1)
        elif not by_category_2 is None:
            L = get_samples_list_by_category_2(by_category_2)
        else:
            L = [i for i in self.sm_matrix]

        return L

    # ---------------- get_samples_list_by_category_1() ----------------
    # 抽取正例样本集，及未标注样本集
    def get_samples_list_by_category_1(self, category_positive):
        positive_samples_list = []
        unlabeled_samples_list = []

        for sample_id in self.sm_matrix:
            (category_id, sample_terms, term_map) = self.sm_matrix[sample_id]
            is_positive = False
            if category_id == category_positive:
                is_positive = True
            else:
                category_1_id = Categories.get_category_1_id(category_id)
                if category_1_id == category_positive:
                    is_positive = True

            #logging.debug("category_id:%d category_positive:%d is_positive:%d" % (category_id, category_positive, is_positive) )
            if is_positive:
                positive_samples_list.append(sample_id)
            else:
                unlabeled_samples_list.append(sample_id)

        return positive_samples_list, unlabeled_samples_list


    # ---------------- get_samples_list_by_category_2() ----------------
    def get_samples_list_by_category_2(self, category_positive):
        positive_samples_list = []
        unlabeled_samples_list = []

        category_1 = Categories.get_category_1_id(category_positive)

        for sample_id in self.sm_matrix:
            (category_id, sample_terms, term_map) = self.sm_matrix[sample_id]

            #if category_id == 2054000:
                #logging.debug("category_id:%d category_positive:%d " % (category_id, category_positive) )

            category_1_id = Categories.get_category_1_id(category_id)
            if category_1_id != category_1:
                continue

            is_positive = False
            if category_id == category_positive:
                is_positive = True

            #logging.debug("category_id:%d category_positive:%d is_positive:%d" % (category_id, category_positive, is_positive) )

            if is_positive:
                positive_samples_list.append(sample_id)
            else:
                unlabeled_samples_list.append(sample_id)

        return positive_samples_list, unlabeled_samples_list

    # ---------------- crossvalidation_by_category_1() ----------------
    def crossvalidation_by_category_1(self, category_1_id, positive_ratio):
        positive_samples_list, unlabeled_samples_list = self.get_samples_list_by_category_1(category_1_id)
        logging.debug("One category - Positive:%d Unlabeled:%d" % (len(positive_samples_list), len(unlabeled_samples_list)))
        n_P_in_U = int(len(positive_samples_list) * (1 - positive_ratio))
        if len(positive_samples_list) > 0:
            if positive_ratio < 1.0:
                n = 0
                while n < n_P_in_U:
                    idx = random.randint(0, len(positive_samples_list) - 1)
                    unlabeled_samples_list.append(positive_samples_list[idx])
                    del positive_samples_list[idx]
                    n += 1
        total_P = len(positive_samples_list)
        total_U = len(unlabeled_samples_list)
        logging.debug("CrossValidation - ratio: %.3f P:%d U:%d P_in_U: %d U_in_U: %d" % (positive_ratio, total_P, total_U, n_P_in_U, total_U - n_P_in_U))
        return positive_samples_list, unlabeled_samples_list


