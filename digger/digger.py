#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    Digger - Digger for positive and unlabeled data.
    异常检测器用于发现在线流式数据中的异常模式，多用于正样本和负样本比例极度失衡的情况，
    如垃圾邮件过滤、舆情负面主题过滤、网络入侵检测等。通常正样本占比极小，无法用常规的
    二类分类技术有效地分离。相关技术包括逻辑回归模型(LR Model)、松弛在线支持向量机(ROSVM)、
    单类支持向量机(OCSVM)、支持向量数据描述(SVDD)、非凸在线支持向量机(LASVM-NC)等。
'''

from datetime import datetime
import argparse
import logging
from logger import Logger, AppWatch
from utils import AppArgs
from os import path
from globals import *

from corpus import Corpus, Samples
from pu_learning import PULearning_test, test_corpus
from mc_learning import multicategories_train, multicategories_predict
#from sne import sne
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

# ---------------- do_rebuild() ----------------
def do_rebuild(corpus_dir, samples_name):
    corpus = Corpus(corpus_dir)
    corpus.vocabulary.load()
    samples = Samples(corpus, samples_name)
    logging.debug(Logger.debug("Rebuild base data..."))
    samples.rebuild()

# ---------------- do_show() ----------------
def do_show(corpus_dir, samples_name):
    if corpus_dir is None or samples_name is None:
        return

    corpus = Corpus(corpus_dir)
    corpus.vocabulary.load()

    samples = Samples(corpus, samples_name)
    samples.load()

    samples.show()


# ---------------- do_export_samples() ----------------
def do_export_samples(corpus_dir, samples_name, xls_file):
    corpus = Corpus(corpus_dir)
    samples = Samples(corpus, samples_name)
    samples.export_samples(xls_file)


# ---------------- do_query_keywords() ----------------
def do_query_keywords(corpus_dir, samples_name, result_dir):
    corpus = Corpus(corpus_dir)

    samples = Samples(corpus, samples_name)
    samples.load()

    samples.show_category_keywords(result_dir)
    #samples.show_keywords_matrix()
    logging.info(Logger.info("Query keywords %s/<%s> Done. %s" % (corpus_dir, samples_name, result_dir)))


# ---------------- do_iem() ----------------
from reliable_negatives import test_iem
def do_iem(corpus_dir, positive_name, unlabeled_name, result_dir):
    test_iem(corpus_dir, positive_name, unlabeled_name, result_dir)


# ---------------- do_sem() ----------------
from reliable_negatives import test_sem
def do_sem(corpus_dir, positive_name, unlabeled_name, result_dir):
    test_sem(corpus_dir, positive_name, unlabeled_name, result_dir)

# ---------------- do_sem() ----------------
def do_pulearning(corpus_dir, positive_name, unlabeled_name, result_dir):
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


# ---------------- do_export_urls() ----------------
def do_export_urls(corpus_dir, samples_name, xls_file):
    corpus = Corpus(corpus_dir)
    samples = Samples(corpus, samples_name)
    samples.export_urls(xls_file)

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

# ---------------- do_query_sample() ----------------
def do_query_sample(corpus_dir, samples_name, sample_id):
    corpus = Corpus(corpus_dir)
    corpus.vocabulary.load()

    samples = Samples(corpus, samples_name)
    samples.load()
    samples.query_by_id(sample_id)


# ---------------- do_query_sample_by_pu() ----------------
def do_query_sample_by_pu(corpus_dir, positive_name, unlabeled_name, sample_id):
    corpus = Corpus(corpus_dir)

    #samples_positive = None
    #for positive_name in positive_name_list:
        #samples = Samples(corpus, positive_name)
        #samples.load()
        #if samples_positive is None:
            #samples_positive = samples
        #else:
            #samples_positive.merge(samples)
            #samples = None

    samples_positive = Samples(corpus, positive_name)
    samples_positive.load()
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


# ---------------- do_refresh() ----------------
def do_refresh(corpus_dir, samples_name):
    corpus = Corpus(corpus_dir)

    samples = Samples(corpus, samples_name)
    samples.load()
    Fix(samples).refresh_content()

# ---------------- do_purge() ----------------
def do_purge(corpus_dir, samples_name):
    corpus = Corpus(corpus_dir)

    samples = Samples(corpus, samples_name)
    samples.load()

    Fix(samples).purge()

# ---------------- do_sne() ----------------
def do_sne(corpus_dir, samples_name, result_dir):
    pass
    #corpus = Corpus(corpus_dir)

    #samples = Samples(corpus, samples_name)
    #samples.load()

    #sne(samples, result_dir, include_null_samples = False)



# ---------------- cmd_import_samples() ----------------
def cmd_import_samples(aa):
    corpus_dir = aa.get_arg('global', 'corpus_dir')
    samples_name = aa.get_arg('global', 'samples_name')

    xls_file = aa.get_arg('exchange', 'xls_file')

    do_import_samples(corpus_dir, samples_name, xls_file)


# ---------------- cmd_rebuild() ----------------
def cmd_rebuild(aa):
    corpus_dir = aa.get_arg('global', 'corpus_dir')
    samples_name = aa.get_arg('global', 'samples_name')
    aa.print_args()
    logging.debug(Logger.debug("corpus_dir: %s samples_name: %s" % (corpus_dir, samples_name)))

    do_rebuild(corpus_dir, samples_name)


# ---------------- cmd_show() ----------------
def cmd_show(aa):
    corpus_dir = aa.get_arg('global', 'corpus_dir')
    samples_name = aa.get_arg('global', 'samples_name')

    do_show(corpus_dir, samples_name)


# ---------------- cmd_query_sample() ----------------
def cmd_query_sample(aa):
    corpus_dir = aa.get_arg('global', 'corpus_dir')
    samples_name = aa.get_arg('global', 'samples_name')
    sample_id = aa.get_arg('query_sample', 'sample_id')
    #do_query_sample(corpus_dir, samples_name, sample_id)

    positive_name = aa.get_arg('PULearning', 'positive_name')
    unlabeled_name = aa.get_arg('PULearning', 'unlabeled_name')
    do_query_sample_by_pu(corpus_dir, positive_name, unlabeled_name, sample_id)


# ---------------- cmd_test() ----------------
def cmd_test(aaargs):
    pass


# ---------------- cmd_export_samples() ----------------
def cmd_export_samples(aa):
    corpus_dir = aa.get_arg('global', 'corpus_dir')
    samples_name = aa.get_arg('global', 'samples_name')

    xls_file = aa.get_arg('exchange', 'xls_file')

    do_export_samples(corpus_dir, samples_name, xls_file)


# ---------------- cmd_query_keywords() ----------------
def cmd_query_keywords(aa):
    corpus_dir = aa.get_arg('global', 'corpus_dir')
    samples_name = aa.get_arg('global', 'samples_name')
    result_dir = aa.get_arg('global', 'result_dir')

    do_query_keywords(corpus_dir, samples_name, result_dir)


# ---------------- cmd_iem() ----------------
def cmd_iem(aa):
    corpus_dir = aa.get_arg('global', 'corpus_dir')
    result_dir = aa.get_arg('global', 'result_dir')

    positive_name = aa.get_arg('PULearning', 'positive_name')
    unlabeled_name = aa.get_arg('PULearning', 'unlabeled_name')

    do_iem(corpus_dir, positive_name, unlabeled_name, result_dir)


# ---------------- cmd_sem() ----------------
def cmd_sem(aa):
    corpus_dir = aa.get_arg('global', 'corpus_dir')
    result_dir = aa.get_arg('global', 'result_dir')

    positive_name = aa.get_arg('PULearning', 'positive_name')
    unlabeled_name = aa.get_arg('PULearning', 'unlabeled_name')

    do_sem(corpus_dir, positive_name, unlabeled_name, result_dir)


# ---------------- cmd_pulearning() ----------------
def cmd_pulearning(aa):
    corpus_dir = aa.get_arg('global', 'corpus_dir')
    result_dir = aa.get_arg('global', 'result_dir')

    positive_name = aa.get_arg('PULearning', 'positive_name')
    unlabeled_name = aa.get_arg('PULearning', 'unlabeled_name')

    do_pulearning(corpus_dir, positive_name, unlabeled_name, result_dir)


# ---------------- update_args() ----------------
def update_args(aa, args):
    # global options
    if hasattr(args, 'corpus_dir'):
        aa.update_arg('corpus_dir', args.corpus_dir)
    if hasattr(args, 'samples_name'):
        aa.update_arg('samples_name', args.samples_name)
    if hasattr(args, 'result_dir'):
        aa.update_arg('result_dir', args.result_dir)

    # xls_file options
    if hasattr(args, 'xls_file'):
        aa.update_arg('xls_file', args.xls_file, section='exchange')

    # PULearning options
    if hasattr(args, 'positive_name'):
        aa.update_arg('positive_name', args.positive_name, section='PULearning')
    if hasattr(args, 'unlabeled_name'):
        aa.update_arg('unlabeled_name', args.unlabeled_name, section='PULearning')

    # Query sample
    if hasattr(args, 'sample_id'):
        aa.update_arg('sample_id', args.sample_id, section='query_sample')

# ---------------- main() ----------------
def main():
    parser = argparse.ArgumentParser(description='Positive and Unlabeled Extractor 1.0')
    parser.add_argument('--corpus_dir', type=str, help='Corpus root dir.')
    parser.add_argument('--samples_name', type=str, help='The samples\'s name in corpus.')
    parser.add_argument('--result_dir', type=str, help='Dir to save result.')

    subparsers = parser.add_subparsers(dest='subcommand', title='subcommands', help='sub-command help')

    # -------- import_samples --------
    parser_import_samples = subparsers.add_parser('import_samples', help='import samples help')
    parser_import_samples.add_argument('--xls_file', type=str, help='The Excel file name will be imported.')
    parser_import_samples.set_defaults(func=cmd_import_samples)

    # -------- rebuild --------
    parser_rebuild = subparsers.add_parser('rebuild', help='rebuild help')
    parser_rebuild.set_defaults(func=cmd_rebuild)

    # -------- show --------
    parser_show = subparsers.add_parser('show', help='show help')
    parser_show.set_defaults(func=cmd_show)

    # -------- query_sample --------
    parser_query_sample = subparsers.add_parser('query_sample', help='query sample help')
    parser_query_sample.add_argument('--sample_id', type=int, help='The sample id.')
    parser_query_sample.set_defaults(func=cmd_query_sample)

    # -------- test --------
    parser_test = subparsers.add_parser('test', help='test help')
    parser_test.set_defaults(func=cmd_test)

    # -------- export_samples --------
    parser_export_samples = subparsers.add_parser('export_samples', help='export samples help')
    parser_export_samples.add_argument('--xls_file', type=str, help='The Excel file name will be imported.')
    parser_export_samples.set_defaults(func=cmd_export_samples)

    # -------- query_keywords --------
    parser_query_keywords = subparsers.add_parser('query_keywords', help='query keywords help')
    parser_query_keywords.set_defaults(func=cmd_query_keywords)

    # -------- iem --------
    parser_iem = subparsers.add_parser('iem', help='IEM help')
    parser_iem.add_argument('--positive_name', type=str, help='The positive samples\'s name in corpus.')
    parser_iem.add_argument('--unlabeled_name', type=str, help='The unlabeled samples\'s name in corpus.')
    parser_iem.set_defaults(func=cmd_iem)

    # -------- sem --------
    parser_sem = subparsers.add_parser('sem', help='SEM help')
    parser_sem.add_argument('--positive_name', type=str, help='The positive samples\'s name in corpus.')
    parser_sem.add_argument('--unlabeled_name', type=str, help='The unlabeled samples\'s name in corpus.')
    parser_sem.set_defaults(func=cmd_sem)

    # -------- pulearning --------
    parser_pulearning = subparsers.add_parser('pulearning', help='PULearning help')
    parser_pulearning.add_argument('--positive_name', type=str, help='The positive samples\'s name in corpus.')
    parser_pulearning.add_argument('--unlabeled_name', type=str, help='The unlabeled samples\'s name in corpus.')
    parser_pulearning.set_defaults(func=cmd_pulearning)

    args = parser.parse_args()
    print args

    aa = AppArgs(['/etc/diggerd/diggerrc', '~/.diggerrc', './.diggerrc'])
    update_args(aa, args)

    aa.write_to_file('./.diggerrc')

    args.func(aa)


if __name__ == '__main__':
    appwatch = AppWatch()
    main()
    appwatch.stop()

