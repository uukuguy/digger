#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''

pu_learning.py Positive and Unlabeled Learner.

'''

from __future__ import division
import logging
from datetime import datetime

from corpus import *
from vocabulary import *
from feature_selection import select_features_by_positive_degree
from utils import *
import positive_degree as pd


def PULearning_test(samples_positive, samples_unlabeled):

    threshold_pd_word = 1.0
    threshold_specialty = 0.8
    threshold_popularity = 0.01

    tsm_positive = samples_positive.tsm
    tsm_unlabeled = samples_unlabeled.tsm

    terms_positive_degree = select_features_by_positive_degree(tsm_positive, tsm_unlabeled, (threshold_pd_word, threshold_specialty, threshold_popularity))

    vocabulary = samples_positive.corpus.vocabulary

    pd.save_terms_positive_degree(terms_positive_degree, vocabulary, "./result/keywords.txt")

    #samples_positive_degree_P = pd.calculate_samples_positive_degree(tsm_positive, terms_positive_degree, max_terms = 20)
    #pd.save_samples_positive_degree(samples_positive, samples_positive_degree_P)

    samples_positive_degree_U = pd.calculate_samples_positive_degree(tsm_unlabeled, terms_positive_degree, max_terms = 20)
    pd.save_samples_positive_degree(samples_unlabeled, samples_positive_degree_U)

    #export_train_svm_file("./result/train.svm", terms_positive_degree, tm_positive, sample_V_P, sample_V_U, tm_unlabeled)
    #export_predict_svm_file("./result/predict.svm", terms_positive_degree, tm_positive, sample_V_P, sample_V_U, tm_unlabeled, samples_unlabeled)


def export_train_svm_file(svm_file, terms_positive_degree, tm_positive, sample_V_P, sample_V_U, tm_unlabeled):
    f = open(svm_file, "wb+")

    for (sample_id, (category, term_map)) in tm_positive.matrix:
        if term_map is None:
            continue
        category = 1
        f.write("%d " % (category))
        terms_list = sorted_dict(term_map, reverse = False)
        for (term_id, term_count) in terms_list:
            #f.write("%d:%d " % (term_id, 1))
            if not term_id in terms_positive_degree:
                continue
            (pd_word, specialty, popularity) = terms_positive_degree[term_id]
            f.write("%d:%.6f " % (term_id, pd_word))
        f.write("\n")


    rowidx = 0
    sample_V_U_list = sorted_dict_by_values(sample_V_U, reverse = False)
    for (sample_id, (V, term_map)) in sample_V_U_list:
        if rowidx >= tm_positive.get_rows():
            break
        if term_map is None:
            continue

        #(_, (category, _)) = tm_unlabeled.matrix[sample_id]
        category = -1
        f.write("%d " % (category))

        terms_list = sorted_dict(term_map, reverse = False)
        for (term_id, term_count) in terms_list:
            #f.write("%d:%d " % (term_id, 1))
            if not term_id in terms_positive_degree:
                continue
            (pd_word, specialty, popularity) = terms_positive_degree[term_id]
            f.write("%d:%.6f " % (term_id, PDword))
        f.write("\n")

        rowidx += 1

    f.close()


def export_predict_svm_file(svm_file, terms_positive_degree, tm_positive, sample_V_P, sample_V_U, tm_unlabeled, samples):
    f0 = open("./result/sample_id.txt", "wb+")
    f = open(svm_file, "wb+")

    rowidx = 0
    sample_V_U_list = sorted_dict_by_values(sample_V_U, reverse = True)
    for (sample_id, (V, term_map)) in sample_V_U_list:
        if rowidx >= tm_unlabeled.get_rows() - tm_positive.get_rows():
            break
        if term_map is None:
            rowidx += 1
            continue
        (_, _, date, title, key, _) = samples.get_sample_meta(sample_id)
        f0.write("%s %s %s\n" % (key, date, title))

        #(_, (category, _)) = tm_unlabeled.matrix[sample_id]
        category = -1
        f.write("%d " % (category))
        terms_list = sorted_dict(term_map, reverse = False)
        for (term_id, term_count) in terms_list:
            #f.write("%d:%d " % (term_id, 1))
            if not term_id in terms_positive_degree:
                continue
            (pd_word, specialty, popularity) = terms_positive_degree[term_id]
            f.write("%d:%.6f " % (term_id, PDword))
        f.write("\n")

        rowidx += 1

    f.close()
    f0.close()


def build_samples_from_xls_file(corpus, samples_name, xls_file):
    samples = Samples(corpus, samples_name)
    samples.import_content_from_xls(xls_file)


def import_test_data(corpus_dir):
    logging.debug("Building corpus %s ..." % (corpus_dir))
    corpus = Corpus(corpus_dir)

    build_samples_from_xls_file(corpus, "po2014_neg_1Q", "./data/po2014_neg_1Q.xls")
    build_samples_from_xls_file(corpus, "po2014_neg_2Q", "./data/po2014_neg_2Q.xls")
    build_samples_from_xls_file(corpus, "po2014_neg_3Q", "./data/po2014_neg_3Q.xls")
    build_samples_from_xls_file(corpus, "po2014_neg_4Q", "./data/po2014_neg_4Q.xls")

    build_samples_from_xls_file(corpus, "po201401", "./data/po201401_full.xls")
    build_samples_from_xls_file(corpus, "po201402", "./data/po201402_full.xls")
    build_samples_from_xls_file(corpus, "po201403", "./data/po201403_full.xls")
    build_samples_from_xls_file(corpus, "po201404", "./data/po201404_full.xls")
    build_samples_from_xls_file(corpus, "po201405", "./data/po201405_full.xls")
    build_samples_from_xls_file(corpus, "po201406", "./data/po201406_full.xls")
    build_samples_from_xls_file(corpus, "po201407", "./data/po201407_full.xls")
    build_samples_from_xls_file(corpus, "po201408", "./data/po201408_full.xls")
    build_samples_from_xls_file(corpus, "po201409", "./data/po201409_full.xls")
    build_samples_from_xls_file(corpus, "po201410", "./data/po201410_full.xls")
    build_samples_from_xls_file(corpus, "po201411", "./data/po201411_full.xls")
    build_samples_from_xls_file(corpus, "po201412", "./data/po201412_full.xls")

def rebuild_test_data(corpus_dir):
    corpus = Corpus(corpus_dir)
    samples_list = corpus.get_samples_list()
    for samples_name in samples_list:
        logging.debug("Rebuild samples %s ..." % (samples_name))
        samples = Samples(corpus, samples_name)
        samples.rebuild()
        samples = None

def test_corpus(corpus_dir, positive_name, unlabeled_name, model_file, svm_file):
    logging.debug("Building corpus %s ..." % (corpus_dir))
    corpus = Corpus(corpus_dir)

    #corpus.export_svm_file("2014_neg_1Q", "po2014_neg_1Q.svm")
    #corpus.export_svm_file("2014_neg_2Q", "po2014_neg_2Q.svm")
    #corpus.export_svm_file("2014_neg_2Q", "po2014_neg_3Q.svm")
    #corpus.export_svm_file("2014_neg_3Q", "po2014_neg_4Q.svm")


    samples_positive = Samples(corpus, positive_name)
    samples_positive.load()
    samples_unlabeled = Samples(corpus, unlabeled_name)
    samples_unlabeled.load()

    PULearning(samples_positive, samples_unlabeled)

if __name__ == '__main__':
    s_time = datetime.utcnow()

    corpus_dir = "po2014.corpus"
    #positive_name = "2014_neg_1Q"
    #unlabeled_name = "po201405"
    #model_file = "po2014.model"
    #svm_file = "po2014.svm"
    #test_corpus(corpus_dir, positive_name, unlabeled_name, model_file, svm_file)

    #import_test_data(corpus_dir)

    rebuild_test_data(corpus_dir)

    e_time = datetime.utcnow()
    t_time = (e_time - s_time)
    logging.info("Done.(%s)" % (str(t_time)))

