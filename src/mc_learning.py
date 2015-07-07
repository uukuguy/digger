#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''

mc_learning.py Multi categories learning.

'''

from __future__ import division
import logging
import os

from utils import sorted_dict
from category_feature_matrix import CategoryFeatureMatrix
from sample_feature_matrix import SampleFeatureMatrix
from classifier import Classifier
from categories import Categories


# ---------------- multicategories_train() ----------------
def multicategories_train(samples_train, model_name = None, result_dir = None):
    if model_name is None:
        model_name = samples_train.name
    if result_dir is None:
        cfm_file = "%s.cfm" % (model_name)
        sfm_file = "%s.sfm" % (model_name)
    else:
        if not os.path.isdir(result_dir):
            try:
                os.mkdir(result_dir)
            except OSError:
                logging.error("mkdir %s failed." % (result_dir))
                return
        cfm_file = "%s/%s.cfm" % (result_dir, model_name)
        sfm_file = "%s/%s.sfm" % (result_dir, model_name)

    cfm, sfm = samples_train.get_categories_1_weight_matrix()
    cfm.save(cfm_file)
    sfm.save(sfm_file)


# ---------------- multicategories_predict() ----------------
def multicategories_predict(samples_test, model_name, result_dir):
    if model_name is None or len(model_name) == 0:
        logging.warn("model_name must not be NULL.")
        return

    if result_dir is None:
        cfm_file = "%s.cfm" % (model_name)
        sfm_file = "%s.sfm" % (model_name)
    else:
        if not os.path.isdir(result_dir):
            try:
                os.mkdir(result_dir)
            except OSError:
                logging.error("mkdir %s failed." % (result_dir))
                return
        cfm_file = "%s/%s.cfm" % (result_dir, model_name)
        sfm_file = "%s/%s.sfm" % (result_dir, model_name)

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

    for sample_id in samples_test.tsm.sample_matrix():
        (sample_category, sample_terms, term_map) = samples_test.tsm.get_sample_row(sample_id)

        category_1_id = Categories.get_category_1_id(sample_category)

        sfm_test.set_sample_category(sample_id, category_1_id)
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

    categories = samples_test.get_categories()

    categories_1_names = []

    categories_1_idx_map = {}
    categories_1_idlist = categories.get_categories_1_idlist()
    for category_id in categories_1_idlist:
        category_idx = sfm_test.get_category_idx(category_id)
        category_name = categories.get_category_name(category_id)
        categories_1_idx_map[category_idx] = (category_id, category_name)
    categories_1_idx_list = sorted_dict(categories_1_idx_map)
    for (category_idx, (category_id, category_name)) in categories_1_idx_list:
        categories_1_names.append("%s(%d)" % (category_name, category_id))

    clf.predict(X_test, y_test, categories_1_names)

