#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''

positive_degree.py 正例度(Positive Degree)

'''

from __future__ import division
import logging
from logger import Logger
from utils import *

# ---------------- calculate_term_speciality() ----------------
def calculate_term_speciality(term_id, tsm_positive, tsm_unlabeled):
    tm_positive = tsm_positive.tm_matrix
    tm_unlabeled = tsm_unlabeled.tm_matrix

    if term_id in tm_positive:
        (_, (term_used_P, _, _)) = tm_positive[term_id]
    else:
        term_used_P = 0
    tf_P = term_used_P / tsm_positive.total_terms_used
    #logging.debug(Logger.debug("Speciality_P: %d/%d" % (term_used_P, tsm_positive.total_terms_used)))

    tf_U = 0.0
    if term_id in tm_unlabeled:
        (_, (term_used_U, _, _)) = tm_unlabeled[term_id]
        tf_U = term_used_U / tsm_unlabeled.total_terms_used
        #logging.debug(Logger.debug("Speciality_U: %d/%d" % (term_used_U, tsm_unlabeled.total_terms_used)))
    speciality = tf_P / (tf_P + tf_U)

    return speciality


# ---------------- calculate_term_popularity() ----------------
def calculate_term_popularity(term_id, tsm):
    sm_matrix = tsm.sm_matrix
    tm_matrix = tsm.tm_matrix

    if not term_id in tm_matrix:
        return 0.0

    (_, (term_used, term_samples, sample_map)) = tm_matrix[term_id]

    ent = 0.0

    for sample_id in sm_matrix:
        if not sample_id in sample_map:
            continue

        # 当前样本sample_id中terms出现的总次数
        (_, sample_terms_0, _) = sm_matrix[sample_id]

        tf_in_samples = term_used
        tf_in_sample = sample_map[sample_id]

        prob_term_in_sample = tf_in_sample / tf_in_samples

        X_sum = 0.0
        for sample_id_1 in sm_matrix:
            (_, sample_terms_1, term_map_1) = sm_matrix[sample_id_1]
            if term_id in term_map_1:
                # X : 指定term在当前样本sample_id_1中出现的次数
                X = sample_map[sample_id_1]
                X_sum += X / tf_in_samples / sample_terms_1
        nprob = prob_term_in_sample / sample_terms_0 / X_sum

        ent += nprob * math.log(nprob)

    total_samples = tsm.get_total_samples()
    if total_samples != 1:
        Z = math.log(total_samples)
        popularity = -ent / Z
    else:
        popularity = 0.0

    return popularity


# ---------------- calculate_term_positive_degree() ----------------
def calculate_term_positive_degree(term_id, tsm_positive, tsm_unlabeled, sensitive_terms):
    if not sensitive_terms is None:
        if term_id in sensitive_terms:
            pd_word = sensitive_terms[term_id]
            speciality = 1.0
            popularity = 1.0
            #selected_terms[term_id] = (pd_word, speciality, popularity)
            return (pd_word, speciality, popularity)

    # -------- Speciality --------
    speciality = calculate_term_speciality(term_id, tsm_positive, tsm_unlabeled)

    # -------- Popularity --------
    popularity = calculate_term_popularity(term_id, tsm_positive)

    # -------- pd_word --------
    #pd_word = speciality + popularity
    pd_word = speciality * popularity

    return (pd_word, speciality, popularity)


# ---------------- calculate_samples_positive_degree() ----------------
def calculate_samples_positive_degree(tsm, terms_positive_degree, max_terms = 20):
    samples_positive_degree = {}
    for sample_id in tsm.sample_matrix():
        samples_positive_degree[sample_id] = (0.0, None)
        (category, sample_terms, term_map) = tsm.get_sample_row(sample_id)
        if sample_terms != 0.0 and sample_terms != 1.0:
            terms_V = {}
            for term_id in term_map:
                if term_id in terms_positive_degree:
                    (pd_word, speciality, popularity) = terms_positive_degree[term_id]
                    terms_V[term_id] = pd_word
            terms_V_list = sorted_dict_by_values(terms_V)
            V = 0.0
            term_idx = 0
            terms_positive_degree_map = {}
            for (term_id, pd_word) in terms_V_list:
                if term_idx >= max_terms:
                    break
                terms_positive_degree_map[term_id] = terms_positive_degree[term_id]
                V += pd_word
                term_idx += 1
            samples_positive_degree[sample_id] = (V / math.log(sample_terms), terms_positive_degree_map)

    return samples_positive_degree


# ---------------- save_terms_positive_degree() ----------------
def save_terms_positive_degree(terms_positive_degree, vocabulary, filename):

    f = open(filename, "wb+")
    terms_positive_degree_list = sorted_dict_by_values(terms_positive_degree, reverse = True)
    for (term_id, (pd_word, speciality, popularity)) in terms_positive_degree_list:
        try:
            term_text = vocabulary.get_term_text(term_id)
        except KeyError:
            term_text = "<KeyError>"
        f.write("%d %s %.6f(%.6f,%.6f)\n" % (term_id, term_text.encode('utf-8'), pd_word, speciality, popularity))
    f.close()


# ---------------- save_samples_positive_degree() ----------------
def save_samples_positive_degree(samples, samples_positive_degree):
    samples_name = samples.name
    output_file = "./result/check_%s.txt" % (samples_name)
    output_full_file = "./result/check_%s_full.txt" % (samples_name)
    f = open(output_file, "wb+")
    f1 = open(output_full_file, "wb+")
    vocabulary = samples.corpus.vocabulary
    rowidx = 0
    sample_V_list = sorted_dict_by_values(samples_positive_degree, reverse = True)
    for (sample_id, (V, term_positive_degree_map)) in sample_V_list:

        if term_positive_degree_map is None:
            rowidx += 1
            continue

        (sample_id_1, category_1, date_1, title, key, url, _) = samples.get_sample_meta(sample_id)

        f.write("[%d] %d %s %d %.6f %s\n" % (rowidx, sample_id, key.encode('utf-8'), category_1, V, title.encode('utf-8')))
        f1.write("[%d] %d %s %d %.6f %s\n" % (rowidx, sample_id, key.encode('utf-8'), category_1, V, title.encode('utf-8')))

        f1.write("    Key Words: \n")
        sample_terms_list = sorted_dict_by_values(term_positive_degree_map, reverse = True)
        words_cnt = 0
        for (term_id, (pd_word, speciality, popularity)) in sample_terms_list:
            if words_cnt >= 10:
                break
            try:
                term_text = vocabulary.get_term_text(term_id)
            except KeyError:
                term_text = "<KeyError>"
            f1.write("    %s - %.6f(%.6f, %.6f)\n" % (term_text.encode('utf-8'), pd_word, speciality, popularity))
            words_cnt += 1
        f1.write("\n")


        if rowidx % 1000 == 0:
            logging.debug(Logger.debug("Check unlabeled samples %d/%d %s" % (rowidx, len(samples_positive_degree), title)))
        rowidx += 1
    f.close()
    f1.close()

