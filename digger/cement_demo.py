#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''

Positive and Unlabeled Extractor.

Usage:
  pue.py test [--corpus_dir=<corpus_dir>] [--positive_name=<pn>] [--unlabeled_name=<un>] [--model_file=<model_file>] [--svm_file=<svm_file>]
  pue.py (-h | --help)
  pue.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --corpus_dir=<corpus_dir>  Corpus root dir [default: po2014.corpus].
  --positive_name=<pn>  Name of the positive samples [default: 2014_neg_1Q].
  --unlabeled_name=<un>  Name of the unlabeled samples [default: po201405].
  --model_file=<model_file> The learning model file name.
  --svm_file=<svm_file> The exported svm file.
'''

'''
    Positive and Unlabeled data Extractor
    异常检测器用于发现在线流式数据中的异常模式，多用于正样本和负样本比例极度失衡的情况，
    如垃圾邮件过滤、舆情负面主题过滤、网络入侵检测等。通常正样本占比极小，无法用常规的
    二类分类技术有效地分离。相关技术包括逻辑回归模型(LR Model)、松弛在线支持向量机(ROSVM)、
    单类支持向量机(OCSVM)、支持向量数据描述(SVDD)、非凸在线支持向量机(LASVM-NC)等。
APP_NAME = 'pue'
'''

from datetime import datetime
from cement.core import backend, foundation, controller, handler, hook
from globals import *

'''
  P - Positive samples
  M - Mixed samples
  RN - Reliable Negative samples
  U - Unlabeled samples
'''
from pu_learner import test_corpus
def default_test(app_args):
    test_corpus(app_args.corpus_dir, app_args.positive_name, app_args.unlabeled_name, app_args.model_file, app_args.svm_file)

def test_1():
    logging.debug("Building corpus...")
    corpus = Corpus("po2014.corpus")

    logging.debug("Building sample1...")
    sample1 = Samples(corpus, "201401")

    #logging.debug("import from xls...")
    #sample1.import_content_from_xls('./data/po201401_full.xls')

    #logging.debug("Rebuild term matrix...")
    #sample1.rebuild_term_matrix()

    #logging.debug("Loding content...")
    #sample1.load_content()

    #logging.debug("Loding term matrix...")
    #tm = sample1.load_term_matrix()

    #logging.debug("Transorm TFIDF")
    #tm_tfidf = tranform_tfidf(tm)

    #logging.debug("Save tfidf matrix...")
    #sample1.save_tfidf_matrix(tm_tfidf)


    #logging.debug("Loading tfidf matrix...")
    #tm_tfidf = sample1.load_tfidf_matrix()

    #save_term_matrix_as_svm_file(tm_tfidf, "po201401_tfidf_3.svm")

    #logging.debug("Loading from svm file...")
    #tm_tfidf = load_from_svm_file("po201401_tfidf.svm")
    #X, y = load_svmlight_file("po201401_tfidf.svm")

class MyBaseApp(foundation.CementApp):
    def __init__(self, label = None, **kw):
        super(MyBaseApp, self).__init__(**kw)

    def log_trace(self, msg):
        self.log.debug("\033[1;34m%s\033[0m" % (msg))

    def log_debug(self, msg):
        self.log.debug("\033[1;37m%s\033[0m" % (msg))

    def log_info(self, msg):
        self.log.info("\033[1;32m%s\033[0m" % (msg))

    def log_notice(self, msg):
        self.log.info("\033[1;36m%s\033[0m" % (msg))

    def log_warn(self, msg):
        self.log.warn("\033[1;35m%s\033[0m" % (msg))

    def log_error(self, msg):
        self.log.error("\033[1;31m%s\033[0m" % (msg))

class MyAppBaseController(controller.CementBaseController):
    class Meta:
        label = 'base'

        config_defaults = dict(
                corpus_dir = ''
                )
        #arguments_override_config = True
        arguments = [
                (['-C', '--corpus_dir'], dict(action='store', help='Corpus root dir')),
                (['-P', '--positive_name'], dict(action='store', help='Positive samples name')),
                (['-U', '--unlabeled_name'], dict(action='store', help='Unlabeled samples name')),
                (['-o', '--model_file'], dict(action='store', help='model file name')),
                (['-s', '--svm_file'], dict(action='store', help='svm file name'))
                ]

    @controller.expose(hide=True, aliases=['run'])
    def default(self):
        self.app.log_info("default")

    # ---------------- test ----------
    @controller.expose(help="Testing.")
    def test(self):
        print "test"
        default_test(app_args)

    # ---------------- train ----------
    @controller.expose(help="Training.")
    def train(self):
        print "train"

    # ---------------- predict ----------
    @controller.expose(help="Predicting.")
    def predict(self):
        print "predict"

    # ---------------- export ----------
    @controller.expose(help="Predicting.")
    def export(self):
        print "export"

class MyApp(MyBaseApp):
    class Meta:
        label = 'MyApp'
        base_controller = MyAppBaseController
        #config_files = ['']

    def __init__(self, label=None, **kw):
        super(MyApp, self).__init__(**kw)

    def _init_args(self):
        app_args.corpus_dir = self.pargs.corpus_dir
        app_args.positive_name = self.pargs.positive_name
        app_args.unlabeled_name = self.pargs.unlabeled_name
        app_args.model_file = self.pargs.model_file
        app_args.svm_file = self.pargs.svm_file


    def hook_post_argument_parsing(self):
        self._init_args()

    def _clean_app(self):
        pass
        #if not self.detector is None:
            #self.detector = None

    def hook_pre_close(self):
        self._clean_app()

def hook_post_argument_parsing(app):
    app.hook_post_argument_parsing()

def hook_pre_close(app):
    app.hook_pre_close()

def main_1():
    s_time = datetime.utcnow()

    app = MyApp(APP_NAME)

    hook.register('post_argument_parsing', hook_post_argument_parsing)
    hook.register('pre_close', hook_pre_close)

    try:
        app.setup()
        app.run()
    finally:
        e_time = datetime.utcnow()
        t_time = (e_time - s_time)
        app.log_notice("Done.(%s)" % (str(t_time)))
        app.close()

from docopt import docopt
def main():
    s_time = datetime.utcnow()

    args = docopt(__doc__, version="Positive and Unlabeled Extractor 1.0")
    #print args
    corpus_dir = args['--corpus_dir']
    positive_name = args['--positive_name']
    unlabeled_name = args['--unlabeled_name']
    model_file = args['--model_file']
    svm_file = args['--svm_file']
    if args['test']:
        test_corpus(corpus_dir, positive_name, unlabeled_name, model_file, svm_file)

    e_time = datetime.utcnow()
    t_time = (e_time - s_time)
    logging.notice("Done.(%s)" % (str(t_time)))

if __name__ == '__main__':
    main()

