#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
feature_selection.py 特征选择算法

'''

from __future__ import division
import math
import logging
from logger import Logger
from term_sample_model import TermSampleModel
from positive_degree import calculate_term_positive_degree


# ---------------- select_features_by_positive_degree() ----------------
def select_features_by_positive_degree(tsm_positive, tsm_unlabeled, (threshold_pd_word, threshold_speciality, threshold_popularity)):

    total_terms = tsm_positive.get_total_terms()
    total_samples = tsm_positive.get_total_samples()

    selected_terms = {}

    rowidx = 0
    logging.debug(Logger.debug("Calculate PDword. %d samples, %d terms in tsm_positive" % (total_samples, total_terms)))
    for term_id in tsm_positive.term_matrix():
        term_info = tsm_positive.get_term_row(term_id)

        (pd_word, speciality, popularity) = calculate_term_positive_degree(term_id, tsm_positive, tsm_unlabeled, None)
        if pd_word >= threshold_pd_word and speciality >= threshold_speciality and popularity >= threshold_popularity:
            selected_terms[term_id] = (pd_word, speciality, popularity)

        if rowidx % 1000 == 0:
            logging.debug(Logger.debug("feature_selection() %d/%d - pd_word:%.6f speciality:%.6f popularity:%.6f" % (rowidx, total_terms, speciality + popularity, speciality, popularity)))
        rowidx += 1

    return selected_terms


# ---------------- get_terms_positive_degree_by_category() ----------------
def get_terms_positive_degree_by_category(tsm, positive_samples_list, unlabeled_samples_list):
    tsm_positive = tsm.clone(positive_samples_list)
    tsm_unlabeled = tsm.clone(unlabeled_samples_list)

    threshold_pd_word = 1.0
    threshold_speciality = 0.8
    threshold_popularity = 0.01
    terms_positive_degree = select_features_by_positive_degree(tsm_positive, tsm_unlabeled, (threshold_pd_word, threshold_speciality, threshold_popularity))

    return terms_positive_degree


