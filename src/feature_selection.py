#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
feature_selection.py 特征选择算法

'''

from __future__ import division
import logging
import math
import logging
from term_sample_matrix import TermSampleMatrix


def calculate_term_specialty(term_id, sm_positive, sm_unlabeled, tsm_positive, tsm_unlabeled):
    #print sm_positive[term_id]
    if term_id in sm_positive:
        #print sm_positive.__class__
        (_, (term_used_P, _, _)) = sm_positive[term_id]
    else:
        term_used_P = 0
    tf_P = term_used_P / tsm_positive.total_terms_used
    #logging.debug("Specialty_P: %d/%d" % (term_used_P, tsm_positive.total_terms_used))

    tf_U = 0.0
    if term_id in sm_unlabeled:
        (_, (term_used_U, _, _)) = sm_unlabeled[term_id]
        tf_U = term_used_U / tsm_unlabeled.total_terms_used
        #logging.debug("Specialty_U: %d/%d" % (term_used_U, tsm_unlabeled.total_terms_used))
    specialty = tf_P / (tf_P + tf_U)

    return specialty

def calculate_term_popularity(term_id, tm_matrix, sm_matrix):
    if not term_id in sm_matrix:
        return 0.0

    (_, (term_used, term_samples, sample_map)) = sm_matrix[term_id]

    ent = 0.0

    for sample_id in tm_matrix:
        if not sample_id in sample_map:
            continue

        (_, sample_terms_0, _) = tm_matrix[sample_id]

        tf_in_samples = term_used
        tf_in_sample = sample_map[sample_id]

        prob_term_in_sample = tf_in_sample / tf_in_samples

        X_sum = 0.0
        for sample_id_1 in tm_matrix:
            (_, sample_terms_1, term_map_1) = tm_matrix[sample_id_1]
            if term_id in term_map_1:
                X = sample_map[sample_id_1]
                X_sum += X / tf_in_samples / sample_terms_1
        nprob = prob_term_in_sample / sample_terms_0 / X_sum

        ent += nprob * math.log(nprob)

    total_samples = len(tm_matrix)
    if total_samples != 1:
        Z = math.log(total_samples)
        popularity = -ent / Z
    else:
        popularity = 0.0

    return popularity


def calculate_term_positive_degree(term_id, tsm_positive, tsm_unlabeled, sensitive_terms):
    if not sensitive_terms is None:
        if term_id in sensitive_terms:
            pd_word = sensitive_terms[term_id]
            specialty = 1.0
            popularity = 1.0
            #selected_terms[term_id] = (pd_word, specialty, popularity)
            return (pd_word, specialty, popularity)

    tm_positive = tsm_positive.tm_matrix
    sm_positive = tsm_positive.sm_matrix

    tm_unlabeled = tsm_unlabeled.tm_matrix
    sm_unlabeled = tsm_unlabeled.sm_matrix

    # -------- Specialty --------
    specialty = calculate_term_specialty(term_id, sm_positive, sm_unlabeled, tsm_positive, tsm_unlabeled)

    # -------- Popularity --------
    tm_matrix = tm_positive
    sm_matrix = sm_positive

    popularity = calculate_term_popularity(term_id, tm_matrix, sm_matrix)

    # -------- pd_word --------
    pd_word = specialty + popularity

    return (pd_word, specialty, popularity)


def select_features_by_positive_degree(tsm_positive, tsm_unlabeled, (threshold_pd_word, threshold_specialty, threshold_popularity)):

    tm_positive = tsm_positive.tm_matrix
    sm_positive = tsm_positive.sm_matrix

    tm_unlabeled = tsm_unlabeled.tm_matrix
    sm_unlabeled = tsm_unlabeled.sm_matrix


    selected_terms = {}

    rowidx = 0
    logging.debug("Calculate PDword. %d in sm_positive" % len(sm_positive))
    for term_id in sm_positive:
        (pd_word, specialty, popularity) = calculate_term_positive_degree(term_id, tsm_positive, tsm_unlabeled, None)
        if pd_word >= threshold_pd_word and specialty >= threshold_specialty and popularity >= threshold_popularity:
            selected_terms[term_id] = (pd_word, specialty, popularity)

        if rowidx % 1000 == 0:
            logging.debug("feature_selection() %d/%d - pd_word:%.6f specialty:%.6f popularity:%.6f" % (rowidx, len(sm_positive), specialty + popularity, specialty, popularity))
        rowidx += 1

    return selected_terms


# ---------------- get_terms_positive_degree_by_category() ----------------
def get_terms_positive_degree_by_category(tsm, positive_samples_list, unlabeled_samples_list):
    tsm_positive = tsm.clone(positive_samples_list)
    tsm_unlabeled = tsm.clone(unlabeled_samples_list)

    threshold_pd_word = 1.0
    threshold_specialty = 0.8
    threshold_popularity = 0.01
    terms_positive_degree = select_features_by_positive_degree(tsm_positive, tsm_unlabeled, (threshold_pd_word, threshold_specialty, threshold_popularity))

    return terms_positive_degree


