#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
feature_weighting.py 特征权值算法

'''
from __future__ import division
import logging
import math
from term_sample_matrix import TermSampleMatrix
from sample_feature_matrix import SampleFeatureMatrix

class FeatureWeight():
    TFIDF = 0
    TFRF = 1
    TFIPNDF = 2

    @staticmethod
    def transform(tsm, fw_type, sfm = None, feature_weights = None):
        if fw_type == FeatureWeight.TFIDF:
            sfm = FeatureWeight.transform_tfidf(tsm, sfm, feature_weights)
        elif fw_type == FeatureWeight.TFRF:
            sfm = FeatureWeight.transform_tfrf(tsm, sfm, feature_weights)
        elif fw_type == FeatureWeight.TFIPNDF:
            sfm = FeatureWeight.transform_tfipndf(tsm, sfm, feature_weights)

        return sfm


    # ---------------- tranform_tfidf() ----------
    @staticmethod
    def transform_tfidf(tsm, sfm = None, feature_weights = None):

        sm_matrix = tsm.sm_matrix
        tm_matrix = tsm.tm_matrix

        if sfm is None:
            logging.debug("transform_tfidf() sfm is None.")
            sfm = SampleFeatureMatrix()
        total_samples = len(sm_matrix)

        #num_samples = sfm.get_num_samples()
        #num_features = sfm.get_num_features()
        #num_categories = sfm.get_num_categories()
        #logging.debug("Before transform_tfidf() sfm: %d samples %d terms %d categories." % (num_samples, num_features, num_categories))

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

        num_samples = sfm.get_num_samples()
        num_features = sfm.get_num_features()
        num_categories = sfm.get_num_categories()
        logging.debug("sfm: %d samples %d terms %d categories." % (num_samples, num_features, num_categories))

        return sfm


    # ---------------- tranform_tfrf() ----------
    @staticmethod
    def transform_tfrf(tsm, sfm = None, feature_weights = None):
        sfm = None
        return sfm


    # ---------------- tranform_tfipndf() ----------
    @staticmethod
    def transform_tfipndf(tsm, sfm = None, feature_weights = None):
        sfm = None
        return sfm


