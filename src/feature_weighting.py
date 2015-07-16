#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
feature_weighting.py 特征权值算法

'''
from __future__ import division
import logging
import math
from term_sample_model import TermSampleModel
from sample_feature_matrix import SampleFeatureMatrix

class FeatureWeight():
    TFIDF = 0
    TFRF = 1
    TFIPNDF = 2

    @staticmethod
    def transform(tsm, sfm, fw_type, feature_weights = None):

        logging.debug("FeatureWeight.transform() tsm: %d samples %d terms." % (tsm.get_total_samples(), tsm.get_total_terms()))

        if sfm is None:
            sfm = SampleFeatureMatrix()
            sfm.init_cagegories(tsm.get_categories())

        if fw_type == FeatureWeight.TFIDF:
            sfm = FeatureWeight.transform_tfidf(tsm, sfm, feature_weights)
        elif fw_type == FeatureWeight.TFRF:
            sfm = FeatureWeight.transform_tfrf(tsm, sfm, feature_weights)
        elif fw_type == FeatureWeight.TFIPNDF:
            sfm = FeatureWeight.transform_tfipndf(tsm, sfm, feature_weights)

        num_samples = sfm.get_num_samples()
        num_features = sfm.get_num_features()
        num_categories = sfm.get_num_categories()
        logging.debug("FeatureWeight.transform(). sfm: %d samples %d terms %d categories." % (num_samples, num_features, num_categories))

        return sfm


    # ---------------- tranform_tfidf() ----------
    @staticmethod
    def transform_tfidf(tsm, sfm, feature_weights = None):

        sm_matrix = tsm.sm_matrix
        tm_matrix = tsm.tm_matrix

        total_samples = len(sm_matrix)

        #num_samples = sfm.get_num_samples()
        #num_features = sfm.get_num_features()
        #num_categories = sfm.get_num_categories()
        #logging.debug("Before transform_tfidf() sfm: %d samples %d terms %d categories." % (num_samples, num_features, num_categories))

        #if feature_weights is None:
        if True:
            rowidx = 0
            for sample_id in sm_matrix:
                (category, sample_terms, term_map) = sm_matrix[sample_id]
                #if category == 1:
                    #print "sample_id: %d category: %d" % (sample_id, category)

                sfm.set_sample_category(sample_id, category)
                colidx = 0
                for term_id in term_map:
                    if not feature_weights is None:
                        if not term_id in feature_weights:
                            #print "sample %d term %d not in feature_weights." % (sample_id, term_id)
                            colidx += 1
                            continue
                    term_used = term_map[term_id]
                    tf = term_used / sample_terms
                    (_, (_, term_samples, _)) = tm_matrix[term_id]
                    idf = math.log(total_samples/term_samples)
                    tfidf = tf * idf
                    sfm.add_sample_feature(sample_id, term_id, tfidf)
                    #print "sample_id: %d term_id: %d tf: %.6f idf: %.6f tfidf: %.6f" % (sample_id, term_id, tf, idf, tfidf)
                    colidx += 1

                rowidx += 1
        else:
            for sample_id in tsm.sample_matrix():
                (category_id, sample_terms, term_map) = tsm.get_sample_row(sample_id)
                sfm.set_sample_category(sample_id, category_id)
                for term_id in term_map:
                    if term_id in feature_weights:
                        feature_weight = feature_weights[term_id]
                        sfm.add_sample_feature(sample_id, term_id, feature_weight)



        return sfm


    # ---------------- tranform_tfrf() ----------
    # rf = log(2 + a / c)
    # a : samples in positive category.
    # c : samples in negative category
    @staticmethod
    def transform_tfrf(tsm, sfm, feature_weights = None):

        if feature_weights is None:
            for term_id in tsm.term_matrix():
                (_, (term_used, term_samples, sample_map)) = tsm.get_term_row(term_id)
                term_categories = {}
                for sample_id in sample_map:
                    category_id = tsm.get_sample_category(sample_id)
                    terms_used_in_sample = sample_map[sample_id]
                    if category_id in term_categories:
                        term_categories[category_id] += terms_used_in_sample
                    else:
                        term_categories[category_id] = terms_used_in_sample
                a = 0
                if 1 in term_categories:
                    a = term_categories[1]
                c = 0
                if -1 in term_categories:
                    c = term_categories[-1]
                if c != 0:
                    rf = math.log(2 + a / c + 1)
                else:
                    rf = math.log(2 + a / 1)


            for sample_id in tsm.sample_matrix():
                (category_id, sample_terms, term_map) = tsm.get_sample_row(sample_id)
                sfm.set_sample_category(sample_id, category_id)
                for term_id in term_map:
                    term_used = term_map[term_id]
                    tf = term_used / sample_terms
                    rf = sfm.feature_weights[term_id]
                    tfrf = tf * rf
                    sfm.add_sample_feature(sample_id, term_id, tfrf)

        else:
            for sample_id in tsm.sample_matrix():
                (category_id, sample_terms, term_map) = tsm.get_sample_row(sample_id)
                sfm.set_sample_category(sample_id, category_id)
                for term_id in term_map:
                    if term_id in feature_weights:
                        feature_weight = feature_weights[term_id]
                        sfm.add_sample_feature(sample_id, term_id, feature_weight)

        return sfm


    # ---------------- tranform_tfipndf() ----------
    @staticmethod
    def transform_tfipndf(tsm, sfm, feature_weights = None):
        return sfm


