#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
reliable_negatives.py - 可信负例样本抽取算法
'''

from __future__ import division
import logging
from logger import Logger
import gmpy2
from sklearn.cluster import KMeans
from utils import sorted_dict, sorted_dict_by_values, crossvalidation_list_by_ratio
from term_sample_model import TermSampleModel
from sample_feature_matrix import SampleFeatureMatrix
from categories import Categories
from feature_weighting import FeatureWeight
from feature_selection import select_features_by_positive_degree
from classifier import Classifier

class BayesianClassifier():
    def __init__(self):
        self.Pr_c = {}
        self.Pr_x_in_c = {}
        self.Pr_f_in_c = {}

    def set_c_prob(self, c, v):
        self.Pr_c[c] = v

    def get_c_prob(self, c):
        if c in self.Pr_c:
            return self.Pr_c[c]
        else:
            return 0.0

    def set_x_in_c_prob(self, c, x, v):
        if x in self.Pr_x_in_c:
            Pr = self.Pr_x_in_c[x]
        else:
            Pr = {}
        Pr[c] = v
        self.Pr_x_in_c[x] = Pr

    def get_x_in_c_prob(self, c, x):
        if x in self.Pr_x_in_c:
            Pr = self.Pr_x_in_c[x]
            if c in Pr:
                return Pr[c]
        return 0.0

    def set_f_in_c_prob(self, c, f, v):
        if f in self.Pr_f_in_c:
            Pr = self.Pr_f_in_c[f]
        else:
            Pr ={}
        Pr[c] = v
        self.Pr_f_in_c[f] = Pr

    def get_f_in_c_prob(self, c, f):
        if f in self.Pr_f_in_c:
            Pr = self.Pr_f_in_c[f]
            if c in Pr:
                return Pr[c]
        return 0.0


    def init_sample_is_category_probs(self, positive_samples_list, negative_samples_list):
        self.Pr_x_in_c = {}
        for sample_id in positive_samples_list:
            self.set_x_in_c_prob(1, sample_id, gmpy2.mpfr(1.0))
            self.set_x_in_c_prob(-1, sample_id, gmpy2.mpfr(0.0))
        for sample_id in negative_samples_list:
            self.set_x_in_c_prob(1, sample_id, gmpy2.mpfr(0.0))
            self.set_x_in_c_prob(-1, sample_id, gmpy2.mpfr(1.0))


    # ---------------- update_feature_in_category_prob() ----------
    def update_feature_in_category_prob(self, tsm):
        categories = tsm.get_categories()
        self.Pr_f_in_c = {}
        for category_id in categories:
            term_prob_sum = gmpy2.mpfr(0.0)
            term_probs = []
            total_terms = tsm.get_total_terms()
            for term_id in tsm.term_matrix():
                (_, (_, _, sample_map)) = tsm.get_term_row(term_id)

                term_prob = gmpy2.mpfr(0.0)
                for sample_id in sample_map:
                    term_used_in_sample = sample_map[sample_id]
                    term_prob += term_used_in_sample * self.get_x_in_c_prob(category_id, sample_id)

                term_probs.append((term_id, term_prob))
                term_prob_sum += term_prob

            for (term_id, term_prob) in term_probs:
                prob = (1.0 + term_prob) / (total_terms + term_prob_sum)

                self.set_f_in_c_prob(category_id, term_id, prob)

    # ---------------- update_category_prob() ----------
    def update_category_prob(self, tsm):
        categories = tsm.get_categories()

        category_probs = {}
        for category_id in categories:
            sample_prob_sum = gmpy2.mpfr(0.0)
            for sample_id in self.Pr_x_in_c:
                sample_prob = self.get_x_in_c_prob(category_id, sample_id)
                sample_prob_sum += sample_prob
            category_probs[category_id] = sample_prob_sum

        #print "self.Pr_f_in_c"
        #print sorted_dict(self.Pr_f_in_c)

        max_prob = gmpy2.mpfr(0.0)
        category_sum = {}
        for term_id in self.Pr_f_in_c:
            x = self.Pr_f_in_c[term_id]
            for category_id in x:
                c_prob = x[category_id]
                #if c_prob > 0.001:
                    #print term_id, category_id, c_prob
                if c_prob > max_prob:
                    max_prob = c_prob
                if category_id in category_sum:
                    category_sum[category_id] += c_prob
                else:
                    category_sum[category_id] = c_prob
        for category_id in category_sum:
            print "category_sum: %d %.6f" % (category_id, category_sum[category_id])

        print "max_prob: %.6f" % (max_prob)

        total_samples = tsm.get_total_samples()
        for category_id in category_probs:
            sample_prob_sum = category_probs[category_id]
            self.set_c_prob(category_id, sample_prob_sum / total_samples)

        print "self.Pr_c:"
        print sorted_dict(self.Pr_c)


    # ---------------- update_feature_and_category_prob() ----------
    # self.Pr_x_in_c -> self.Pr_f_in_c
    def update_feature_and_category_prob(self, tsm):
        self.update_feature_in_category_prob(tsm)
        self.update_category_prob(tsm)


    # ---------------- compute_sample_is_category_prob() ----------
    # self.Pr_f_in_c -> self.Pr_x_in_c
    # 更新所有未标记样本的后验概率Pr[c1|di]
    def compute_sample_is_category_prob(self, tsm):
        categories = tsm.get_categories()
        for sample_id in tsm.sample_matrix():
            # 所有正例样本的分类概率保持不变

            sample_prob_sum = gmpy2.mpfr(0.0)
            sample_probs = []
            for category_id in categories:
                Pr_c = self.get_c_prob(category_id)

                sample_prob_base = gmpy2.mpfr(1.0)
                (_, _, term_map) = tsm.get_sample_row(sample_id)
                for term_id in term_map:
                    term_prob = self.get_f_in_c_prob(category_id, term_id)
                    sample_prob_base *= term_prob

                sample_prob = sample_prob_base * Pr_c
                sample_probs.append((category_id, sample_prob))
                sample_prob_sum += sample_prob

            for (category_id, sample_prob) in sample_probs:
                prob = sample_prob / sample_prob_sum
                self.set_x_in_c_prob(category_id, sample_id, prob)


    # ---------------- result() ----------
    def result(self):
        sample_categories = {}
        for sample_id in self.Pr_x_in_c:
            x_probs = self.Pr_x_in_c[sample_id]

            max_prob = gmpy2.mpfr(0.0)
            likely_category = None
            idx = 0
            #print sample_id, x_probs
            for category_id in x_probs:
                prob = x_probs[category_id]
                if idx == 0:
                    max_prob = prob
                    likely_category = category_id
                else:
                    if prob > max_prob:
                        max_prob = prob
                        likely_category = category_id
                idx += 1
            if not likely_category is None:
                sample_categories[sample_id] = (likely_category, max_prob, x_probs)

            #print "result() - sample_id: %d positive: %.6f negative: %.6f likely_category: %d" % (sample_id, x_probs[1], x_probs[-1], likely_category)

        return sample_categories


    # ---------------- EM() ----------------
    def EM(self, tsm_train, tsm_test):
        clf = self
        sample_categories = {}
        n = 0
        FP0 = FN0 = 0
        predict_result = {}
        while True:
            logging.debug(Logger.debug("-- %d -- Predicting ..." % (n)))
            sc = clf.predict(tsm_test)
            #print sc
            TP, TN, FP, FN = report_em_result(tsm_test, sc)

            if FP == 0:
                break
            if FP0 > 0 and FP > FP0:
                break
            if FP == FP0 and FN <= FN0:
                break

            new_predict_result = {}
            for sample_id in sc:
                (likely_category, prob, x_probs) = sc[sample_id]
                new_predict_result[sample_id] = likely_category
            if new_predict_result == predict_result:
                break
            predict_result = new_predict_result

            logging.debug(Logger.debug("-- %d -- Building new NB-C ..." % (n)))
            clf.fit(tsm_train)

            sample_categories = {k:sc[k] for k in sc}

            FP0 = FP
            FN0 = FN
            n += 1

        return sample_categories


    # ---------------- fit() ----------------
    def fit(self, tsm):
        self.update_feature_and_category_prob(tsm)


    # ---------------- predict() ----------------
    def predict(self, tsm):
        self.compute_sample_is_category_prob(tsm)
        return self.result()

class ReliableNegatives():

    # ---------------- __init__() ----------------
    def __init__(self):
        pass

    # ---------------- I_EM() ----------------
    #  category_id = [1, -1] where samples in tsm_P and tsm_M
    def I_EM(self, tsm_P, tsm_M, positive_category_id):

        # -------- tsm_test --------
        tsm_test = tsm_M.clone()
        tsm_test.set_all_samples_target(-1)
        negative_samples_list = tsm_test.get_samples_list()

        # -------- tsm_train --------
        tsm_train = tsm_P.clone()
        tsm_train.set_all_samples_target(1)
        positive_samples_list = tsm_train.get_samples_list()
        tsm_train.merge(tsm_test, renewid = False)

        logging.debug(Logger.debug("ReliableNegatives.I_EM() tsm_train (positive:%d, mixed:%d) has %d samples." % (tsm_P.get_total_samples(), tsm_M.get_total_samples(), tsm_train.get_total_samples())))

        # Build initial Naive Bayesian Classifier NB-C
        logging.debug(Logger.debug("Building init NB-C ..."))
        clf = BayesianClassifier()
        logging.debug(Logger.debug("init_sample_is_category_probs() Positive: %d Negative: %d ..." % (len(positive_samples_list), len(negative_samples_list))))
        clf.init_sample_is_category_probs(positive_samples_list, negative_samples_list)
        clf.fit(tsm_train)

        sample_categories = clf.EM(tsm_train, tsm_test)

        return sample_categories


    # ---------------- sem_step1() ----------------
    def sem_step1(self, tsm_positive, tsm_unlabeled, spy_ratio, spy_threshold_ratio, positive_category_id):
        NS = []
        US = []
        P0 = tsm_positive.get_samples_list()
        S, P = crossvalidation_list_by_ratio(P0, spy_ratio)
        M = tsm_unlabeled.get_samples_list()
        MS = M + S

        # -------- I-EM --------
        tsm_P = tsm_positive.clone(P)
        #for sample_id in tsm_P.sm_matrix:
            #tsm_P.set_sample_category(sample_id, 1)

        tsm_MS = tsm_unlabeled.clone(M)
        tsm_S = tsm_positive.clone(S)
        tsm_MS.merge(tsm_S, renewid = False)
        logging.debug(Logger.debug("tsm_MS(Unlabeled(%d) + Spy(%d)) total samples: %d" % (tsm_unlabeled.get_total_samples(), tsm_S.get_total_samples(), tsm_MS.get_total_samples())))
        #for sample_id in tsm_MS.sm_matrix:
            #tsm_MS.set_sample_category(sample_id, -1)

        tsm_P.set_all_samples_category(1)
        tsm_P.categories = [1, -1]
        for sample_id in tsm_MS.sample_matrix():
            category_id = tsm_MS.get_sample_category(sample_id)
            if Categories.get_category_1_id(category_id) == positive_category_id:
                tsm_MS.set_sample_category(sample_id, 1)
            else:
                tsm_MS.set_sample_category(sample_id, -1)
        tsm_MS.categories = [1, -1]

        sample_categories = self.I_EM(tsm_P, tsm_MS, positive_category_id)

        # -------- Spy positive probability threshold --------
        # 计算spy分类概率阈值t
        Pr_spy = {}
        for sample_id in S:
            if not sample_id in sample_categories:
                logging.warn(Logger.warn("Spy sample %d not in sample_categories." % (sample_id)))
                continue
            spy_category, prob, x_probs = sample_categories[sample_id]
            Pr_spy[sample_id] = x_probs[1]
        Pr_spy_list = sorted_dict_by_values(Pr_spy, reverse = False)

        #spy_idx = 2
        num_spy = len(S)
        spy_idx = int(num_spy * spy_threshold_ratio)
        (sample_id, t) = Pr_spy_list[spy_idx]

        idx = 0
        for (sample_id, spy_prob) in Pr_spy_list:
            spy_category, prob, x_probs = sample_categories[sample_id]
            print idx, sample_id, spy_category, prob, x_probs

            idx += 1

        logging.debug(Logger.debug("Spy sample id: %d spy_threshold: %s (spy_idx=%d)" % (sample_id, str(t), spy_idx)))

        # -------- Reliable Negative Samples --------
        for sample_id in M:
            category_id, prob, x_probs = sample_categories[sample_id]
            prob = x_probs[1]
            #print "sample_id: %d category_id: %d prob: %.3f t: %.3f" % (sample_id, category_id, prob, t)
            if prob < t:
                NS.append(sample_id)
            else:
                US.append(sample_id)

        return NS, US


    # ---------------- sem_step2() ----------------
    def sem_step2(self, tsm_P, tsm_U, PS, NS, US, positive_category_id):

        tsm_positive = tsm_P.clone()
        tsm_unlabeled = tsm_U.clone()

        tsm_positive.set_all_samples_category(1)
        for sample_id in tsm_unlabeled.sample_matrix():
            category_id = tsm_unlabeled.get_sample_category(sample_id)
            if Categories.get_category_1_id(category_id) == positive_category_id:
                tsm_unlabeled.set_sample_category(sample_id, 1)
            else:
                tsm_unlabeled.set_sample_category(sample_id, -1)

        tsm_train = tsm_positive.clone()
        tsm_train.merge(tsm_unlabeled, renewid = False)

        clf = BayesianClassifier()
        logging.debug(Logger.debug("init_sample_is_category_probs() Positive: %d Negative: %d ..." % (len(PS), len(NS))))
        clf.init_sample_is_category_probs(PS, NS)
        clf.fit(tsm_train)

        logging.debug(Logger.debug("-- %d -- Building the final classifier using P, N, U ..." % (n)))
        sample_categories = clf.EM(tsm_train, tsm_unlabeled)

        return sample_categories


# ---------------- calculate_representative_prototype() ----------------
def calculate_representative_prototype(tsm, NS, US):
    tsm_negative = tsm.clone(NS)
    sfm_negative = FeatureWeight.transform(tsm_negative, FeatureWeight.TFIDF)
    sfm_negative.init_cagegories([1,-1])
    X, y = sfm_negative.to_sklearn_data(include_null_samples = False)
    t = 30
    m = int(t * len(NS) / (len(NS) +len(US)))
    logging.debug(Logger.debug("Clustering NS into %d micro-clusters." % (m)))
    est = KMeans(n_clusters = m)
    est.fit(X)
    labels = est.labels_
    print type(labels), len(labels)
    kk = {}
    for n in labels:
        if n in kk:
            kk[n] += 1
        else:
            kk[n] = 1
    print kk
    for k in kk:
        if kk[k] > 5:
            print "----- cluster %d (%d)" % (k, kk[k])
            n = 0
            idx = 0
            for l in labels:
                if n >= 5:
                    break
                if l == k:
                    sample_id = NS[idx]
                    category_id = tsm.get_sample_category(sample_id)
                    print k, sample_id, category_id
                    n += 1
                idx += 1


    #tsm_unlabeled = tsm.clone(US)

# ---------------- rn_iem() ----------------
def rn_iem(positive_category_id, tsm_positive, tsm_unlabeled, result_dir):
    rn = ReliableNegatives()
    sample_categories = rn.I_EM(tsm_positive, tsm_unlabeled, positive_category_id)

# ---------------- do_feature_selection() ----------------
def do_feature_selection(tsm_positive, tsm_other):
    threshold_pd_word = 1.0
    threshold_specialty = 0.8
    threshold_popularity = 0.3
    terms_positive_degree = select_features_by_positive_degree(tsm_positive, tsm_other, (threshold_pd_word, threshold_specialty, threshold_popularity))
    #print terms_positive_degree

    terms_positive_degree_list = sorted_dict_by_values(terms_positive_degree, reverse = True)
    idx = 0
    for (term_id, (pd_word, specialty, popularity)) in terms_positive_degree_list:
        term_text = tsm_positive.vocabulary.get_term_text(term_id)
        print "[%d] %d %s %.6f(%.6f,%.6f)" % (idx, term_id, term_text.encode('utf-8'), pd_word, specialty, popularity)
        if idx >= 30:
            break
        idx += 1

    selected_terms = [term_id for (term_id, _) in terms_positive_degree_list]

    return selected_terms

# ---------------- make_train_test_set() ----------------
def make_train_test_set(tsm_P, tsm_U, PS, NS, US, positive_category_id):
    tsm_positive = tsm_P.clone()
    tsm_unlabeled = tsm_U.clone()

    # Convert multi level categories to binary categories. [1, -1]
    # -------- All positive samples set category to 1
    tsm_positive.set_all_samples_category(1)
    tsm_positive.init_categories([1, -1])
    # -------- Negative samples
    # set category to 1 if it's level-1 category equal positive_category_id,
    # otherwise set category to -1
    p0 = 0
    u0 = 0
    for sample_id in tsm_unlabeled.sample_matrix():
        category_id = tsm_unlabeled.get_sample_category(sample_id)
        if Categories.get_category_1_id(category_id) == positive_category_id:
            tsm_unlabeled.set_sample_category(sample_id, 1)
            p0 += 1
        else:
            tsm_unlabeled.set_sample_category(sample_id, -1)
            u0 += 1
    tsm_unlabeled.init_categories([1, -1])
    logging.debug(Logger.debug("tsm_unlabeled: P: %d U: %d P+U: %d" % (p0, u0, p0 + u0)))


    # -------- reliable negative --------
    diff_NS = list(set(NS).difference(set(PS)))
    tsm_diffns = tsm_unlabeled.clone(diff_NS)
    #RN = []
    #for sample_id in diff_NS:
        #if tsm_unlabeled.get_sample_category(sample_id) == -1:
            #RN.append(sample_id)
    #tsm_diffns = tsm_unlabeled.clone(RN)
    # There are some positive samples in reliable negative set.
    tsm_diffns.set_all_samples_category(-1)


    # -------- feature selection --------
    #L_other = tsm_U.get_samples_list(by_category_1 = positive_category_id, exclude = True)
    #tsm_other = tsm_unlabeled.clone(L_other)
    tsm_other = tsm_diffns
    selected_terms = do_feature_selection(tsm_positive, tsm_other)
    logging.debug(Logger.debug("After do_feature_selection() selected %d terms." % (len(selected_terms))))
    #selected_terms = None

    # -------- tsm_train --------
    tsm_train = tsm_positive.clone(terms_list = selected_terms)
    logging.debug(Logger.debug("tsm_train cloned."))
    if selected_terms is None:
        selected_terms = tsm_train.get_terms_list()
        logging.debug(Logger.debug("selected_terms is None. use tsm_train %d terms" % (len(selected_terms))))
    else:
        logging.debug(Logger.debug("%d selected terms" % (len(selected_terms))))
    tsm_diffns = tsm_diffns.clone(terms_list = selected_terms)
    logging.debug(Logger.debug("tsm_diffns cloned."))
    tsm_train.merge(tsm_diffns, renewid = False)
    logging.debug(Logger.debug("tsm_train merged tsm_diffns. %d samples %d terms" % (tsm_train.get_total_samples(), tsm_train.get_total_terms())))

    # -------- tsm_test --------
    tsm_test = tsm_unlabeled.clone(terms_list = selected_terms)
    logging.debug(Logger.debug("tsm_test cloned using %d selected terms. %d samples %d terms" % (len(selected_terms), tsm_test.get_total_samples(), tsm_test.get_total_terms())))

    return tsm_train, tsm_test


# ---------------- rocsvm() ----------------
def rocsvm(tsm_train, tsm_test):
    fw_type = FeatureWeight.TFIDF
    #fw_type = FeatureWeight.TFRF

    # -------- sfm_train --------
    logging.debug(Logger.debug("rocsvm() transform tsm_train(%d samples %d terms) ..." % (tsm_train.get_total_samples(),  tsm_train.get_total_terms())))

    sfm_train = SampleFeatureMatrix()
    sfm_train = FeatureWeight.transform(tsm_train, sfm_train, fw_type)

    # -------- sfm_test --------
    logging.debug(Logger.debug("rocsvm() transform tsm_test(%d samples %d terms) ..." % (tsm_test.get_total_samples(),  tsm_test.get_total_terms())))

    sfm_test = SampleFeatureMatrix(category_id_map = sfm_train.get_category_id_map(), feature_id_map = sfm_train.get_feature_id_map())
    sfm_test.init_cagegories([1, -1])

    sfm_test = FeatureWeight.transform(tsm_test, sfm_test, fw_type)

    include_null_samples = True
    # -------- train & predict --------
    logging.debug(Logger.debug("rocsvm() to_sklearn_data sfm_train(%d samples %d features %d categories) ..." % (sfm_train.get_num_samples(), sfm_train.get_num_features(), sfm_train.get_num_categories())))
    X_train, y_train = sfm_train.to_sklearn_data(include_null_samples = include_null_samples)

    logging.debug(Logger.debug("rocsvm() to_sklearn_data sfm_test(%d samples %d features %d categories) ..." % (sfm_test.get_num_samples(), sfm_test.get_num_features(), sfm_test.get_num_categories())))
    X_test, y_test = sfm_test.to_sklearn_data(include_null_samples = include_null_samples)

    clf = Classifier()
    clf.train(X_train, y_train)

    y_pred = clf.predict(X_test, y_test, [u"Positive", u"Negative"])

    P_pred = []
    N_pred = []
    test_samples_list = sfm_test.get_samples_list(include_null_samples = include_null_samples)
    idx = 0
    for sample_id in test_samples_list:
        category_id = sfm_test.get_category_id(y_pred[idx])
        if category_id == 1:
            P_pred.append(sample_id)
        else:
            N_pred.append(sample_id)
        idx += 1

    return P_pred, N_pred


# ---------------- rn_sem() ----------------
def rn_sem(positive_category_id, tsm_positive, tsm_unlabeled, result_dir):
    rn = ReliableNegatives()

    spy_ratio = 0.1
    spy_threshold_ratio = 0.15


    best_log = []
    best_log0 = []

    NS_best = []
    US_best = []
    TP = TN = FP = FN = 0
    TP0 = TN0 = FP0 = FN0 = 0
    common_NS = []
    n = 0
    while n < 1:
        NS, US = rn.sem_step1(tsm_positive, tsm_unlabeled, spy_ratio, spy_threshold_ratio, positive_category_id)

        TP, TN, FP, FN = report_sem_result(tsm_positive, tsm_unlabeled, NS, US, positive_category_id)
        best_log.append((TP, TN, FP, FN))

        if n == 0:
            common_NS = NS
        else:
            #if FP == FP0 and FN <= FN0:
                #break
            #common_NS = NS_best
            common_NS = list(set(common_NS).intersection(set(NS)))
            diff_NS = list(set(NS).difference(set(common_NS)))
            US = US + diff_NS
            logging.debug(Logger.debug("======== %d common NS (P:%d, U:%d) ========" % (n, len(common_NS), len(US))))
            TP0, TN0, FP0, FN0 = report_sem_result(tsm_positive, tsm_unlabeled, common_NS, US, positive_category_id)
            best_log0.append((FP0, FN0, UP0, UN0))

        NS_best = [i for i in NS]
        US_best = [i for i in US]
        n += 1

    print " \t| TP\t| TN\t| accu\t| FP\t| FN\t|"
    idx = 0
    for (TP, TN, FP, FN) in best_log:
        if TN + TP > 0.0:
            accu = TN / (TN + FP)
        else:
            accu = 0.0
        print "%d\t| %d\t| %d\t| %.3f\t| %d\t| %d\t|" % (idx, TP, TN, accu, FP, FN)
        idx += 1
    print
    print " \t| FP0\t| FN0\t| accu\t| UP0\t| UN0\t|"
    idx = 0
    for (TP, TN, FP, FN) in best_log0:
        if TN + TP > 0.0:
            accu = TN / (TN + TP)
        else:
            accu = 0.0
        print "%d\t| %d\t| %d\t| %.3f\t| %d\t| %d\t|" % (idx, TP, TN, accu, FP, FN)
        idx += 1
    print

    PS = tsm_positive.get_samples_list()

    # -------- ROC-SVM --------
    tsm_train, tsm_test = make_train_test_set(tsm_positive, tsm_unlabeled, PS, NS_best, US_best, positive_category_id)
    P_pred, N_pred = rocsvm(tsm_train, tsm_test)

    TP = TN = FP = FN = 0
    for sample_id in P_pred:
        category_id = tsm_unlabeled.get_sample_category(sample_id)
        if Categories.get_category_1_id(category_id) == positive_category_id:
            TP += 1
        else:
            TN += 1
    for sample_id in N_pred:
        category_id = tsm_unlabeled.get_sample_category(sample_id)
        if Categories.get_category_1_id(category_id) == positive_category_id:
            FP += 1
        else:
            FN += 1

    show_confusion_matrix(TP, TN, FP, FN)

    # -------- S-EM --------
    #rn.sem_step2(tsm_positive, tsm_unlabeled, PS, NS_best, US_best, positive_category_id)


    #calculate_representative_prototype(tsm_unlabeled, NS_best, US_best)


# ---------------- report_sem_result() ----------------
def report_sem_result(tsm_positive, tsm_unlabeled, NS, US, positive_category_id):
    TN = 0
    TP = 0
    for sample_id in NS:
        category_id = tsm_unlabeled.get_sample_category(sample_id)
        category_1_id = Categories.get_category_1_id(category_id)
        if category_id is None:
            logging.warn(Logger.warn("category_id is None in NS. sample_id: %d positive_category_id: %d" % (sample_id, positive_category_id)))
        if category_1_id != positive_category_id:
            TN += 1
        else:
            TP += 1

    FP = 0
    FN = 0
    for sample_id in US:
        category_id = tsm_unlabeled.get_sample_category(sample_id)
        if category_id is None:
            logging.warn(Logger.warn("category_id is None in US. sample_id: %d positive_category_id: %d" % (sample_id, positive_category_id)))
        category_1_id = Categories.get_category_1_id(category_id)
        if category_1_id == positive_category_id:
            FP += 1
        else:
            FN += 1

    print "Total Reliable Negatives: %d" % (TP + TN)
    print "TP: %d TN: %d" % (TP, TN)
    if TN + TP > 0:
        accuracy = TN / (TN + TP)
    else:
        accuracy = 0.0
    print "Accuracy: %.3f" % (accuracy)
    print
    print "Total Unlabeled: %d" % (FP + FN)
    print "FP: %d FN: %d" % (FP, FN)
    print

    return TP, TN, FP, FN


def show_confusion_matrix(TP, TN, FP, FN):
    print "\t| True\t| False\t|"
    print "Positive| %d\t| %d\t| %d" % (TP, FP, TP + FP)
    print "Negative| %d\t| %d\t| %d" % (TN, FN, TN + FN)
    print "\t| %d\t| %d\t|" % (TP + TN, FP + FN)
    print

    #print "- Positive -"
    print "         \t precision \t recall \t F-score \t support"
    if TP + TN != 0.0:
        positive_accuracy = TP / (TP + TN)
    else:
        positive_accuracy = 0.0
    if TP + FP != 0.0:
        positive_recall = TP / (TP + FP)
    else:
        positive_recall = 0.0
    if positive_accuracy + positive_recall != 0:
        positive_F = 2 * positive_accuracy * positive_recall / (positive_accuracy + positive_recall)
    else:
        positive_F = 0.0
    print "  Positive\t %.3f%% \t %.3f%% \t %.3f%% \t %d" % (positive_accuracy * 100, positive_recall * 100, positive_F * 100, TP + FP)
    #print "Accuracy: %.3f%%" % (positive_accuracy * 100)
    #print "Recall: %.3f%%" % (positive_recall * 100)
    #print

    #print "- Negative -"
    if FN + FP != 0.0:
        negative_accuracy = FN / (FN + FP)
    else:
        negative_accuracy = 0.0
    if FN + TN != 0.0:
        negative_recall = FN / (FN + TN)
    else:
        negative_recall = 0.0
    if negative_accuracy + negative_recall != 0:
        negative_F = 2 * negative_accuracy * negative_recall / (negative_accuracy + negative_recall)
    else:
        negative_F = 0.0
    print "  Negative\t %.3f%% \t %.3f%% \t %.3f%% \t %d" % (negative_accuracy * 100, negative_recall * 100, negative_F * 100, TN + FN)
    print
    #print "Accuracy: %.3f%%" % (negative_accuracy * 100)
    #print "Recall: %.3f%%" % (negative_recall * 100)
    #print

# ---------------- report_em_result() ----------------
def report_em_result(tsm_test, sample_categories):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    #report_cnt = 0
    for sample_id in sample_categories:
        (likely_category_id, prob, x_probs) = sample_categories[sample_id]

        category_id = tsm_test.get_sample_category(sample_id)
        #print category_id, likely_category_id
        if category_id is None:
            continue

        if category_id == 1:
            if likely_category_id == 1:
                TP += 1
            else:
                FP += 1
        else:
            if likely_category_id == -1:
                FN += 1
            else:
                TN += 1
        #logging.debug(Logger.debug("sample_id: %d positive_category_id: %d category_id: %d likely_category: %d TP %d FP %d TN %d FN %d" % (sample_id, positive_category_id, category_id, likely_category_id, TP, FP, TN, FN)))

    show_confusion_matrix(TP, TN, FP, FN)

    return TP, TN, FP, FN

