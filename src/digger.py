#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Digger

Usage:
  digger.py import_samples [--corpus_dir=<corpus_dir>] [--samples_name=<sn>] [--xls_file=<xls_file>]
  digger.py export_samples [--corpus_dir=<corpus_dir>] [--samples_name=<sn>] [--xls_file=<xls_file>]
  digger.py export_urls [--corpus_dir=<corpus_dir>] [--samples_name=<sn>] [--xls_file=<xls_file>]
  digger.py rebuild [--corpus_dir=<corpus_dir>] [--samples_name=<sn>]
  digger.py rebuild_categories [--corpus_dir=<corpus_dir>] [--samples_name=<sn>]
  digger.py test [--corpus_dir=<corpus_dir>] [--positive_name=<pn>] [--unlabeled_name=<un>] [--model_file=<model_file>] [--svm_file=<svm_file>]
  digger.py query_categories [--corpus_dir=<corpus_dir>] [--samples_name=<sn>] [--xls_file=<xls_file>]
  digger.py query_keywords [--corpus_dir=<corpus_dir>] [--samples_name=<sn>] [--result_dir <rd>]
  digger.py query_sample [--corpus_dir=<corpus_dir>] [--positive_name=<pn>] [--unlabeled_name=<un>] [--samples_name=<sn>] [--sample_id=<sid>]
  digger.py refresh [--corpus_dir=<corpus_dir>] [--samples_name=<sn>]
  digger.py show [--corpus_dir=<corpus_dir>] [--samples_name=<sn>]
  digger.py purge [--corpus_dir=<corpus_dir>] [--samples_name=<sn>]
  digger.py sne [--corpus_dir=<corpus_dir>] [--samples_name=<sn>]
  digger.py train [--corpus_dir=<corpus_dir>] [--samples_name=<sn>] [--model_name=<mn>]
  digger.py predict [--corpus_dir=<corpus_dir>] [--samples_name=<sn>] [--model_name=<mn>]
  digger.py iem [--corpus_dir=<corpus_dir>] [--positive_name=<pn>] [--unlabeled_name=<un>] [--result_dir <rd>]
  digger.py sem [--corpus_dir=<corpus_dir>] [--positive_name=<pn>] [--unlabeled_name=<un>] [--result_dir <rd>]
  digger.py (-h | --help)
  digger.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --corpus_dir=<corpus_dir>  Corpus root dir [default: po2013.corpus].
  --result_dir=<rd>  Result output dir [default: ./result].
  --positive_name=<pn>  Name of the positive samples [default: 2014_neg_1Q].
  --unlabeled_name=<un>  Name of the unlabeled samples [default: po201405].
  --model_file=<model_file>  The learning model file name.
  --svm_file=<svm_file>  The exported svm file.
  --samples_name=<sn>  The samples's name in corpus.
  --sample_id=<sid>  The sample's id.
  --xls_file=<xls_file>  The Excel file name will be imported.
  --model_name=<mn>  Model name.

'''

'''
    Digger - Digger for positive and unlabeled data.
    异常检测器用于发现在线流式数据中的异常模式，多用于正样本和负样本比例极度失衡的情况，
    如垃圾邮件过滤、舆情负面主题过滤、网络入侵检测等。通常正样本占比极小，无法用常规的
    二类分类技术有效地分离。相关技术包括逻辑回归模型(LR Model)、松弛在线支持向量机(ROSVM)、
    单类支持向量机(OCSVM)、支持向量数据描述(SVDD)、非凸在线支持向量机(LASVM-NC)等。
'''

from datetime import datetime
from logger import Logger
import logging
from docopt import docopt
from globals import *
from corpus import Corpus, Samples
from pu_learning import PULearning_test, test_corpus
from mc_learning import multicategories_train, multicategories_predict
from sne import sne
from fix import Fix

'''
  P - Positive samples
  M - Mixed samples
  RN - Reliable Negative samples
  U - Unlabeled samples
'''

# ---------------- do_import_samples() ----------------
def do_import_samples(corpus_dir, samples_name, xls_file):
    corpus = Corpus(corpus_dir)
    corpus.vocabulary.load()
    samples = Samples(corpus, samples_name)
    samples.import_samples(xls_file)

# ---------------- do_export_samples() ----------------
def do_export_samples(corpus_dir, samples_name, xls_file):
    corpus = Corpus(corpus_dir)
    samples = Samples(corpus, samples_name)
    samples.export_samples(xls_file)

# ---------------- do_export_urls() ----------------
def do_export_urls(corpus_dir, samples_name, xls_file):
    corpus = Corpus(corpus_dir)
    samples = Samples(corpus, samples_name)
    samples.export_urls(xls_file)

# ---------------- do_rebuild() ----------------
def do_rebuild(corpus_dir, samples_name):
    corpus = Corpus(corpus_dir)
    corpus.vocabulary.load()
    samples = Samples(corpus, samples_name)
    logging.debug(Logger.debug("Rebuild base data..."))
    samples.rebuild()

# ---------------- do_rebuild_categories() ----------------
def do_rebuild_categories(corpus_dir, samples_name):
    corpus = Corpus(corpus_dir)
    samples = Samples(corpus, samples_name)
    samples.load()
    logging.debug(Logger.debug("Rebuild base data..."))
    Fix(samples).fix_categories()
    samples.rebuild_categories()

# ---------------- do_train() ----------------
def do_train(corpus_dir, samples_name, model_name, result_dir):
    corpus = Corpus(corpus_dir)
    corpus.vocabulary.load()
    samples = Samples(corpus, samples_name)
    samples.load()
    logging.debug(Logger.debug("Training ..."))
    multicategories_train(samples, model_name, result_dir)

# ---------------- do_predict() ----------------
def do_predict(corpus_dir, samples_name, model_name, result_dir):
    corpus = Corpus(corpus_dir)
    corpus.vocabulary.load()
    samples = Samples(corpus, samples_name)
    samples.load()
    logging.debug(Logger.debug("Predicting ..."))
    multicategories_predict(samples, model_name, result_dir)

# ---------------- do_iem() ----------------
from reliable_negatives import ReliableNegatives, rn_iem, rn_sem
def do_iem(corpus_dir, positive_name, unlabeled_name, result_dir):
    corpus = Corpus(corpus_dir)
    corpus.vocabulary.load()
    samples_positive = Samples(corpus, positive_name)
    samples_positive.load()
    #samples_unlabeled = Samples(corpus, unlabeled_name)
    #samples_unlabeled.load()

    logging.debug(Logger.debug("I-EM ..."))

    positive_category_id = 4000000
    positive_ratio = 0.8
    tsm = samples_positive.tsm
    positive_samples_list, unlabeled_samples_list = tsm.crossvalidation_by_category_1(positive_category_id, positive_ratio, random = False)

    tsm_positive = tsm.clone(positive_samples_list)
    tsm_unlabeled = tsm.clone(unlabeled_samples_list)

    rn_iem(positive_category_id, tsm_positive, tsm_unlabeled, result_dir)

# ---------------- do_sem() ----------------
from reliable_negatives import ReliableNegatives, rn_iem
def do_sem(corpus_dir, positive_name, unlabeled_name, result_dir):
    corpus = Corpus(corpus_dir)
    corpus.vocabulary.load()
    samples_positive = Samples(corpus, positive_name)
    samples_positive.load()
    #samples_unlabeled = Samples(corpus, unlabeled_name)
    #samples_unlabeled.load()

    logger.debug(Logger.debug("S-EM ..."))

    #positive_category_id = 1000000 # 供电服务
    #positive_category_id = 2000000 # 人资管理
    positive_category_id = 6000000 # 安全生产
    #positive_category_id = 6000000 # 党建作风
    #positive_category_id = 8000000 # 依法治企
    positive_ratio = 0.4
    negative_ratio = 0.66 # ratio of remaing samples. (1 - positive_ratio) * negative_ratio
    tsm = samples_positive.tsm
    #for sample_id in tsm.sample_matrix():
        #category_id = tsm.get_sample_category(sample_id)
        #print sample_id, category_id

    positive_samples_list, unlabeled_samples_list = tsm.crossvalidation_by_category_1(positive_category_id, positive_ratio, negative_ratio, positive_random = False, negative_random = False)

    tsm_positive = tsm.clone(positive_samples_list)
    tsm_unlabeled = tsm.clone(unlabeled_samples_list)

    #print positive_samples_list
    #print unlabeled_samples_list

    total_positive_samples = tsm_positive.get_total_samples()
    total_unlabeled_samples = tsm_unlabeled.get_total_samples()
    logging.debug(Logger.debug("do_sem() %d samples in tsm_positive, %d samples in tsm_unlabeled." % (total_positive_samples, total_unlabeled_samples)))
    #for sample_id in tsm_unlabeled.sample_matrix():
        #category_id = tsm_unlabeled.get_sample_category(sample_id)
        #print sample_id, category_id

    rn_sem(positive_category_id, tsm_positive, tsm_unlabeled, result_dir)


# ---------------- do_test() ----------------
def do_test(corpus_dir, positive_name, unlabeled_name, model_file, svm_file):
    corpus = Corpus(corpus_dir)
    #corpus.vocabulary.load()

    samples_positive = Samples(corpus, positive_name)
    samples_positive.load()
    #samples_positive = None
    #for positive_name in positive_name_list:
        #samples = Samples(corpus, positive_name)
        #samples.load()
        #if samples_positive is None:
            #samples_positive = samples
        #else:
            #samples_positive.merge(samples)
            #samples = None

    samples_unlabeled = Samples(corpus, unlabeled_name)
    samples_unlabeled.load()

    PULearning_test(samples_positive, samples_unlabeled)


# ---------------- do_query_sample() ----------------
def do_query_sample(corpus_dir, samples_name, sample_id):
    corpus = Corpus(corpus_dir)

    samples = Samples(corpus, samples_name)
    samples.load()
    samples.query_by_id(sample_id)


# ---------------- do_query_sample_by_pu() ----------------
def do_query_sample_by_pu(corpus_dir, positive_name_list, unlabeled_name, sample_id):
    corpus = Corpus(corpus_dir)

    samples_positive = None
    for positive_name in positive_name_list:
        samples = Samples(corpus, positive_name)
        samples.load()
        if samples_positive is None:
            samples_positive = samples
        else:
            samples_positive.merge(samples)
            samples = None

    samples_unlabeled = Samples(corpus, unlabeled_name)
    samples_unlabeled.load()

    corpus.query_by_id(samples_positive, samples_unlabeled, sample_id)


# ---------------- do_query_categories() ----------------
def do_query_categories(corpus_dir, samples_name, xls_file):
    corpus = Corpus(corpus_dir)

    samples = Samples(corpus, samples_name)
    samples.load()

    samples.query_categories(xls_file)
    logging.info(Logger.info("Query categories %s/<%s> Done. %s" % (corpus_dir, samples_name, xls_file)))


# ---------------- do_query_keywords() ----------------
def do_query_keywords(corpus_dir, samples_name, result_dir):
    corpus = Corpus(corpus_dir)

    samples = Samples(corpus, samples_name)
    samples.load()

    samples.show_category_keywords(result_dir)
    #samples.show_keywords_matrix()
    logging.info(Logger.info("Query keywords %s/<%s> Done. %s" % (corpus_dir, samples_name, result_dir)))


# ---------------- do_refresh() ----------------
def do_refresh(corpus_dir, samples_name):
    corpus = Corpus(corpus_dir)

    samples = Samples(corpus, samples_name)
    samples.load()
    Fix(samples).refresh_content()

# ---------------- do_show() ----------------
def do_show(corpus_dir, samples_name):
    corpus = Corpus(corpus_dir)
    #corpus.vocabulary.load()

    samples = Samples(corpus, samples_name)
    samples.load()

    samples.show()

# ---------------- do_purge() ----------------
def do_purge(corpus_dir, samples_name):
    corpus = Corpus(corpus_dir)

    samples = Samples(corpus, samples_name)
    samples.load()

    Fix(samples).purge()

# ---------------- do_sne() ----------------
def do_sne(corpus_dir, samples_name, result_dir):
    corpus = Corpus(corpus_dir)

    samples = Samples(corpus, samples_name)
    samples.load()

    sne(samples, result_dir, include_null_samples = False)

# ---------------- main() ----------------
def main():
    s_time = datetime.utcnow()

    args = docopt(__doc__, version="Positive and Unlabeled Extractor 1.0")
    #print args

    corpus_dir = args['--corpus_dir']
    result_dir = args['--result_dir']
    #positive_name_list = args['--positive_name']
    positive_name = args['--positive_name']
    unlabeled_name = args['--unlabeled_name']
    model_file = args['--model_file']
    svm_file = args['--svm_file']
    samples_name = args['--samples_name']
    model_name = args['--model_name']
    arg_sample_id = args['--sample_id']

    if not arg_sample_id is None:
        sample_id = int(arg_sample_id)
    else:
        sample_id = None

    xls_file = args['--xls_file']
    if args['test']:
        do_test(corpus_dir, positive_name, unlabeled_name, model_file, svm_file)
    elif args['import_samples']:
        do_import_samples(corpus_dir, samples_name, xls_file)
    elif args['export_samples']:
        do_export_samples(corpus_dir, samples_name, xls_file)
    elif args['export_urls']:
        do_export_urls(corpus_dir, samples_name, xls_file)
    elif args['rebuild']:
        do_rebuild(corpus_dir, samples_name)
    elif args['rebuild_categories']:
        do_rebuild_categories(corpus_dir, samples_name)
    elif args['query_sample']:
        if not samples_name is None:
            do_query_sample(corpus_dir, samples_name, sample_id)
        else:
            do_query_sample_by_pu(corpus_dir, positive_name_list, unlabeled_name, sample_id)
    elif args['query_categories']:
        do_query_categories(corpus_dir, samples_name, xls_file)
    elif args['query_keywords']:
        do_query_keywords(corpus_dir, samples_name, result_dir)
    elif args['refresh']:
        do_refresh(corpus_dir, samples_name)
    elif args['show']:
        do_show(corpus_dir, samples_name)
    elif args['purge']:
        do_purge(corpus_dir, samples_name)
    elif args['sne']:
        do_sne(corpus_dir, samples_name, result_dir)
    elif args['train']:
        do_train(corpus_dir, samples_name, model_name, result_dir)
    elif args['predict']:
        do_predict(corpus_dir, samples_name, model_name, result_dir)
    elif args['iem']:
        do_iem(corpus_dir, positive_name, unlabeled_name, result_dir)
    elif args['sem']:
        do_sem(corpus_dir, positive_name, unlabeled_name, result_dir)

    e_time = datetime.utcnow()
    t_time = (e_time - s_time)
    logging.info(Logger.info("Done.(%s)" % (str(t_time))))

if __name__ == '__main__':
    main()

