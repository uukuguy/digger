#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''

experiments.py - 试验课题

'''

from __future__ import division
import logging
from logger import Logger
from utils import *

from corpus import *
from vocabulary import *
from feature_selection import select_features_by_positive_degree
import positive_degree as pd

# ---------------- negative_opinion_judge() ----------------
# 负面舆情判定
def negative_opinion_judge(samples_train, samples_test):
    tsm_train = samples_train.tsm
    tsm_test = samples_test.tsm

# ---------------- topic_classification() ----------------
# 议题分类
def topic_classification(samples):
    tsm = samples.tsm

# ---------------- topic_classification() ----------------
# 主题词提取
def topic_keywords_extraction(samples):
    tsm = samples.tsm

