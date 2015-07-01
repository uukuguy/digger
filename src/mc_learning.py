#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''

mc_learning.py Multi categories learning.

'''

from __future__ import division
import logging

from category_feature_matrix import CategoryFeatureMatrix
from sample_feature_matrix import SampleFeatureMatrix
from classifier import Classifier


def multicategories_train(samples_train, model_name = None):
    if model_name is None:
        model_name = samples_train.name
    cfm_file = "%s.cfm" % (model_name)
    sfm_file = "%s.sfm" % (model_name)

    cfm, sfm = samples_train.get_categories_1_weight_matrix()
    cfm.save(cfm_file)
    sfm.save(sfm_file)

def multicategories_predict(samples_test, model_name):
    if model_name is None or len(model_name) == 0:
        logging.warn("model_name must not be NULL.")
        return

    cfm_file = "%s.cfm" % (model_name)
    sfm_file = "%s.sfm" % (model_name)

    logging.debug("Loading train sample feature matrix ...")
    sfm_train = SampleFeatureMatrix()
    sfm_train.load(sfm_file)
    logging.debug("Loading train category feature matrix ...")
    cfm_train = CategoryFeatureMatrix()
    cfm_train.load(cfm_file)

    logging.debug("Making sample feature matrix for test data ...")
    category_id = 2000000
    sfm_test = SampleFeatureMatrix(sfm_train.get_category_id_map(), sfm_train.get_feature_id_map())

    features = cfm_train.get_features(category_id)
    tm_matrix = samples_test.tsm.tm_matrix
    for sample_id in tm_matrix:
        sample_info = tm_matrix[sample_id]
        (sample_category, sample_terms, term_map) = sample_info
        sfm_test.set_sample_category(sample_id, int(sample_category / 1000000) * 1000000)
        for feature_id in features:
            if feature_id in term_map:
                feature_weight = features[feature_id]
                sfm_test.add_sample_feature(sample_id, feature_id, feature_weight)

    logging.debug("train sample feature matrix - features:%d categories:%d" % (sfm_train.get_num_features(), sfm_train.get_num_categories()))
    X_train, y_train = sfm_train.to_sklearn_data()

    logging.debug("test sample feature matrix - features:%d categories:%d" % (sfm_test.get_num_features(), sfm_test.get_num_categories()))
    X_test, y_test = sfm_test.to_sklearn_data()

    clf = Classifier()
    logging.debug("Classifier training ...")
    clf.train(X_train, y_train)
    logging.debug("Classifier predicting ...")
    clf.predict(X_test, y_test)

