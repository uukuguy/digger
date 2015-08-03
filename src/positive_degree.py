#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''

positive_degree.py 正例度(Positive Degree)

'''

from __future__ import division
import logging
from logger import Logger
from utils import *

# ---------------- calculate_samples_positive_degree() ----------------
def calculate_samples_positive_degree(tsm, terms_positive_degree, max_terms = 20):
    samples_positive_degree = {}
    for sample_id in tsm.sample_matrix():
        (category, sample_terms, term_map) = tsm.get_sample_row(sample_id)
        if sample_terms != 0.0 and sample_terms != 1.0:
            terms_V = {}
            for term_id in term_map:
                if term_id in terms_positive_degree:
                    (pd_word, specialty, popularity) = terms_positive_degree[term_id]
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
        else:
            samples_positive_degree[sample_id] = (0.0, None)

    return samples_positive_degree


# ---------------- save_terms_positive_degree() ----------------
def save_terms_positive_degree(terms_positive_degree, vocabulary, filename):

    f = open(filename, "wb+")
    terms_positive_degree_list = sorted_dict_by_values(terms_positive_degree, reverse = True)
    for (term_id, (pd_word, specialty, popularity)) in terms_positive_degree_list:
        try:
            term_text = vocabulary.get_term_text(term_id)
        except KeyError:
            term_text = "<KeyError>"
        f.write("%d %s %.6f(%.6f,%.6f)\n" % (term_id, term_text.encode('utf-8'), pd_word, specialty, popularity))
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

        #print key.__class__
        f.write("[%d] %d %s %d %.6f %s\n" % (rowidx, sample_id, key.encode('utf-8'), category_1, V, title.encode('utf-8')))
        f1.write("[%d] %d %s %d %.6f %s\n" % (rowidx, sample_id, key.encode('utf-8'), category_1, V, title.encode('utf-8')))

        f1.write("    Key Words: \n")
        sample_terms_list = sorted_dict_by_values(term_positive_degree_map, reverse = True)
        words_cnt = 0
        for (term_id, (pd_word, specialty, popularity)) in sample_terms_list:
            if words_cnt >= 10:
                break
            try:
                term_text = vocabulary.get_term_text(term_id)
            except KeyError:
                term_text = "<KeyError>"
            f1.write("    %s - %.6f(%.6f, %.6f)\n" % (term_text.encode('utf-8'), pd_word, specialty, popularity))
            words_cnt += 1
        f1.write("\n")


        if rowidx % 1000 == 0:
            logging.debug(Logger.debug("Check unlabeled samples %d/%d %s" % (rowidx, len(samples_positive_degree), title)))
        rowidx += 1
    f.close()
    f1.close()

