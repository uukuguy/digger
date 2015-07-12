#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
reliable_negatives.py - 可信负例样本抽取算法
'''

from __future__ import division
import logging
from sklearn.cluster import KMeans
from utils import sorted_dict, sorted_dict_by_values, crossvalidation_list_by_ratio
from term_sample_matrix import TermSampleMatrix
from sample_feature_matrix import SampleFeatureMatrix
from categories import Categories
from feature_weighting import FeatureWeight
from classifier import Classifier

class BayesianClassifier():
    def __init__(self, categories):
        self.Pr_c = {}
        self.Pr_x_in_c = {}
        self.Pr_f_in_c = {}
        self.categories = categories

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


    def init_x_in_c_probs(self, positive_sample_list, nagetive_sample_list):
        for sample_id in positive_sample_list:
            self.set_x_in_c_prob(1, sample_id, 1.0)
            self.set_x_in_c_prob(-1, sample_id, 0.0)
        for sample_id in nagetive_sample_list:
            self.set_x_in_c_prob(1, sample_id, 0.0)
            self.set_x_in_c_prob(-1, sample_id, 1.0)


    def __calculate_f_in_c_base(self, tsm, c):
        non_zero_probs = []

        term_prob_sum = 0.0
        term_probs = []
        total_terms = tsm.get_total_terms()
        for term_id in tsm.term_matrix():
            (_, (_, _, sample_map)) = tsm.get_term_row(term_id)

            term_prob = 0.0
            for sample_id in sample_map:
                term_used_in_sample = sample_map[sample_id]
                term_prob += term_used_in_sample * self.get_x_in_c_prob(c, sample_id)

            term_probs.append((term_id, term_prob))
            term_prob_sum += term_prob

        for (term_id, term_prob) in term_probs:
            prob = (1.0 + term_prob) / (total_terms + term_prob_sum)

            self.set_f_in_c_prob(c, term_id, prob)
            if prob > 0.0:
                non_zero_probs.append(prob)
            else:
                print "term_id: %d prob: %.6f term_prob: %.6f term_prob_sum: %.6f total_terms: %d" % (term_id, prob, term_prob, term_prob_sum, total_terms)

            #logging.debug("term_id: %d c: %d prob: %.6f total_terms: %d" % (term_id, c, prob, total_terms))

        prob_sum = 0.0
        non_zero_count = len(non_zero_probs)
        for prob in non_zero_probs:
            prob_sum += prob
        #logging.debug("----------- Feature average Probability: %d %.6f" % (c, prob_sum / non_zero_count))



    # ---------------- calculate_f_in_c_prob() ----------
    # self.Pr_x_in_c -> self.Pr_f_in_c
    def calculate_f_in_c_prob(self, tsm):
        self.Pr_f_in_c = {}
        categories = self.categories
        category_probs = {}
        for category_id in categories:
            self.__calculate_f_in_c_base(tsm, category_id)

            sample_prob_sum = 0.0
            for sample_id in self.Pr_x_in_c:
                sample_prob = self.get_x_in_c_prob(category_id, sample_id)
                sample_prob_sum += sample_prob
            category_probs[category_id] = sample_prob_sum

        #print "self.Pr_f_in_c"
        #print sorted_dict(self.Pr_f_in_c)

        max_prob = 0.0
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


    def __calculate_x_in_c_base(self, tsm, c, sample_id):
        sample_prob = 1.0
        (_, _, term_map) = tsm.get_sample_row(sample_id)
        for term_id in term_map:
            term_prob = self.get_f_in_c_prob(c, term_id)
            if term_prob > 0.0:
                #term_prob *= 10000
                sample_prob *= term_prob

        #if sample_prob == 0.0:
            #sample_prob = 1.0
            #(_, _, term_map) = tsm.get_sample_row(sample_id)
            #for term_id in term_map:
                #term_prob = self.get_f_in_c_prob(c, term_id)
                #if term_prob > 0.0:
                    ##term_prob *= 10000
                    #sample_prob *= term_prob
                ##logging.debug("sample_id: %d term_id:%d term_prob: %.6f sample_prob:%.9f" % (sample_id, term_id, term_prob, sample_prob))

        return sample_prob

    # ---------------- calculate_x_in_c_prob() ----------
    # self.Pr_f_in_c -> self.Pr_x_in_c
    # 更新所有未标记样本的后验概率Pr[c1|di]
    def calculate_x_in_c_prob(self, tsm):
        #total_terms = tsm.get_total_terms()
        #default_term_prob = 1.0 / total_terms
        default_term_prob = 0.5
        categories = self.categories
        for sample_id in tsm.sample_matrix():
            # 所有正例样本的分类概率保持不变

            sample_prob_sum = 0.0
            sample_probs = []
            for category_id in categories:
                Pr_c = self.get_c_prob(category_id)
                sample_prob_base = self.__calculate_x_in_c_base(tsm, category_id, sample_id)
                #if sample_prob_base == 0.0:
                    #logging.warn("calculate_x_in_c_prob() sample_prob_base == 0.0")
                sample_prob = Pr_c * sample_prob_base
                sample_probs.append((category_id, sample_prob))
                sample_prob_sum += sample_prob
                #logging.debug("Pr_c: %.3f sample_prob_base: %.9f" % (Pr_c, sample_prob_base))
            if sample_prob_sum > 0.0:
                for (category_id, sample_prob) in sample_probs:
                    prob = sample_prob / sample_prob_sum
                    self.set_x_in_c_prob(category_id, sample_id, prob)
            else:
                #logging.warn("sample_prob_sum == 0.0")
                self.set_x_in_c_prob(1, sample_id, default_term_prob)
                self.set_x_in_c_prob(-1, sample_id, default_term_prob)
                #self.set_x_in_c_prob(1, sample_id, self.get_c_prob(1))
                #self.set_x_in_c_prob(-1, sample_id, self.get_c_prob(-1))


            #sample_prob_c1 = self.__calculate_x_in_c_base(tsm, 1, sample_id)
            #sample_prob_c2 = self.__calculate_x_in_c_base(tsm, -1, sample_id)
            #logging.debug("c1: %.6f c2: %.6f sample_prob: c1:%.32f c2:%.32f sum:%.32f" % (self.get_c_prob(1), self.get_c_prob(-1), sample_prob_c1, sample_prob_c2, sample_prob_c1 + sample_prob_c2))

        #print "self.Pr_x_in_c"
        #print sorted_dict(self.Pr_x_in_c)

    # ---------------- result() ----------
    def result(self):
        sample_categories = {}
        #print "result() self.Pr_x_in_c:"
        #print self.Pr_x_in_c
        for sample_id in self.Pr_x_in_c:
            x_probs = self.Pr_x_in_c[sample_id]

            max_prob = 0.0
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
                sample_categories[sample_id] = (likely_category, max_prob)

            #print "result() - sample_id: %d positive: %.6f nagetive: %.6f likely_category: %d" % (sample_id, x_probs[1], x_probs[-1], likely_category)

        return sample_categories


class ReliableNegatives():

    # ---------------- __init__() ----------------
    def __init__(self):
        pass

    def __init_samples(self, tsm_P, tsm_U):
        tsm_positive = tsm_P.clone()
        tsm_unlabeled = tsm_U.clone()

        total_unlabeled_samples = tsm_unlabeled.get_total_samples()
        for sample_id in tsm_positive.sample_matrix():
            tsm_positive.set_sample_category(sample_id, 1)
        for sample_id in tsm_unlabeled.sample_matrix():
            tsm_unlabeled.set_sample_category(sample_id, -1)

        tsm = tsm_positive.clone()
        tsm.merge(tsm_unlabeled, renewid = False)

        logging.debug("ReliableNegatives.__init_samples() tsm (positive:%d, unlabeled:%d) has %d samples." % (tsm_positive.get_total_samples(), tsm_unlabeled.get_total_samples(), tsm.get_total_samples()))

        return tsm, tsm_positive, tsm_unlabeled


    # ---------------- I_EM() ----------------
    def I_EM(self, tsm_P, tsm_U, positive_category_id):
        tsm, tsm_positive, tsm_unlabeled= self.__init_samples(tsm_P, tsm_U)
        categories = tsm.get_categories()
        print categories

        predict_result = {}
        positive_sample_list = []
        nagetive_sample_list = []
        for sample_id in tsm.sample_matrix():
            category_id = tsm.get_sample_category(sample_id)
            if category_id == 1:
                positive_sample_list.append(sample_id)
                predict_result[sample_id] = 1
            else:
                nagetive_sample_list.append(sample_id)
                predict_result[sample_id] = -1

        # Build initial Naive Bayesian Classifier NB-C
        clf = BayesianClassifier(categories)
        logging.debug("init_x_in_c_probs() Positive: %d Nagetive: %d ..." % (len(positive_sample_list), len(nagetive_sample_list)))
        clf.init_x_in_c_probs(positive_sample_list, nagetive_sample_list)
        #print clf.Pr_x_in_c

        logging.debug("calculate_f_in_c_prob() ...")
        clf.calculate_f_in_c_prob(tsm)

        sample_categories = None
        n = 0
        while True:
            logging.debug("-- %d -- Predicting ..." % (n))
            clf.calculate_x_in_c_prob(tsm_unlabeled)
            sample_categories = clf.result()
            TP, TN, FP, FN = report_iem_result(tsm_P, tsm_U, sample_categories, positive_category_id)

            if FP == 0:
                break

            new_predict_result = {}
            for sample_id in sample_categories:
                (likely_category, prob) = sample_categories[sample_id]
                new_predict_result[sample_id] = likely_category
            if new_predict_result == predict_result:
                break
            predict_result = new_predict_result

            logging.debug("-- %d -- Building new NB-C ..." % (n))
            clf.calculate_f_in_c_prob(tsm)
            n += 1

        return tsm, sample_categories

    def S_EM(self, tsm_positive, tsm_unlabeled, spy_ratio, spy_threshold_ratio, positive_category_id):
        NS = []
        US = []
        P0 = tsm_positive.get_samples_list()
        S, P = crossvalidation_list_by_ratio(P0, spy_ratio)
        M = tsm_unlabeled.get_samples_list()
        MS = M + S

        tsm_P = tsm_positive.clone(P)
        #for sample_id in tsm_P.sm_matrix:
            #tsm_P.set_sample_category(sample_id, 1)

        tsm_MS = tsm_unlabeled.clone(M)
        tsm_S = tsm_positive.clone(S)
        tsm_MS.merge(tsm_S, renewid = False)
        logging.debug("tsm_MS(Unlabeled(%d) + Spy(%d)) total samples: %d" % (tsm_unlabeled.get_total_samples(), tsm_S.get_total_samples(), tsm_MS.get_total_samples()))
        #for sample_id in tsm_MS.sm_matrix:
            #tsm_MS.set_sample_category(sample_id, -1)

        tsm, sample_categories = self.I_EM(tsm_P, tsm_MS, positive_category_id)

        # 计算spy分类概率阈值t
        Pr_spy = {}
        for sample_id in S:
            spy_category, prob = sample_categories[sample_id]
            if spy_category == 1:
                Pr_spy[sample_id] = prob
            else:
                Pr_spy[sample_id] = 1.0 - prob
        Pr_spy_list = sorted_dict_by_values(Pr_spy, reverse = False)
        num_spy = len(S)
        spy_idx = int(num_spy * spy_threshold_ratio)
        (sample_id, t) = Pr_spy_list[spy_idx]
        print Pr_spy_list
        logging.debug("Spy sample id: %d spy_threshold: %.6f (spy_idx=%d)" % (sample_id, t, spy_idx))

        for sample_id in M:
            category_id, prob = sample_categories[sample_id]
            if category_id == -1:
                prob = 1.0 - prob
            #print "sample_id: %d category_id: %d prob: %.3f t: %.3f" % (sample_id, category_id, prob, t)
            if prob < t:
                NS.append(sample_id)
            else:
                US.append(sample_id)

        return NS, US

def calculate_representative_prototype(tsm, NS, US):
    tsm_nagetive = tsm.clone(NS)
    sfm_nagetive = FeatureWeight.transform(tsm_nagetive, FeatureWeight.TFIDF)
    X, y = sfm_nagetive.to_sklearn_data()
    t = 30
    m = int(t * len(NS) / (len(NS) +len(US)))
    logging.debug("Clustering NS into %d micro-clusters." % (m))
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
    tsm, sample_categories = rn.I_EM(tsm_positive, tsm_unlabeled, positive_category_id)

# ---------------- rocsvm() ----------------
def rocsvm(tsm_P, tsm_U, PS, NS, US, positive_category_id):
    tsm_positive = tsm_P.clone()
    tsm_unlabeled = tsm_U.clone()
    for sample_id in tsm_positive.sample_matrix():
        tsm_positive.set_sample_category(sample_id, 1)
    for sample_id in tsm_unlabeled.sample_matrix():
        category_id = tsm_unlabeled.get_sample_category(sample_id)
        if Categories.get_category_1_id(category_id) == positive_category_id:
            tsm_unlabeled.set_sample_category(sample_id, 1)
        else:
            tsm_unlabeled.set_sample_category(sample_id, -1)

    #tsm = tsm_positive.clone()
    #tsm.merge(tsm_unlabeled, renewid = False)

    #categories = tsm.get_categories()
    #print categories

    #sfm = FeatureWeight.transform(tsm, FeatureWeight.TFIDF)

    tsm_train = tsm_positive.clone()

    diff_NS = list(set(NS).difference(set(PS)))
    tsm_ns = tsm_unlabeled.clone(diff_NS)
    for sample_id in tsm_ns.sample_matrix():
        tsm_ns.set_sample_category(sample_id, -1)
    tsm_train.merge(tsm_ns, renewid = False)

    #sfm_train = SampleFeatureMatrix(sfm.get_category_id_map(), sfm.get_feature_id_map())
    sfm_train = SampleFeatureMatrix()
    sfm_train = FeatureWeight.transform(tsm_train, FeatureWeight.TFIDF, sfm_train)
    #sfm_train = FeatureWeight.transform(tsm_train, FeatureWeight.TFIDF)

    tsm_test = tsm_unlabeled.clone(US)

    sfm_test = SampleFeatureMatrix(feature_weights = sfm_train.feature_weights, category_id_map = sfm_train.get_category_id_map(), feature_id_map = sfm_train.get_feature_id_map())

    #num_samples = sfm_test.get_num_samples()
    #num_features = sfm_test.get_num_features()
    #num_categories = sfm_test.get_num_categories()
    #logging.debug("sfm_test: %d samples %d terms %d categories." % (num_samples, num_features, num_categories))

    sfm_test = FeatureWeight.transform(tsm_test, FeatureWeight.TFIDF, sfm_test, sfm_train.feature_weights)

    #num_samples = sfm_test.get_num_samples()
    #num_features = sfm_test.get_num_features()
    #num_categories = sfm_test.get_num_categories()
    #logging.debug("After transform tfidf. sfm_test: %d samples %d terms %d categories." % (num_samples, num_features, num_categories))

    X_train, y_train = sfm_train.to_sklearn_data()
    X_test, y_test = sfm_test.to_sklearn_data()

    clf = Classifier()
    clf.train(X_train, y_train)

    # predicting

    #clf.predict(X_train, y_train, [u"Nagetive", u"Positive"])
    clf.predict(X_test, y_test, [u"Nagetive", u"Positive"])


# ---------------- sem_last() ----------------
def sem_last(tsm_P, tsm_U, PS, NS, US, positive_category_id):

    tsm_positive = tsm_P.clone()
    tsm_unlabeled = tsm_U.clone()
    for sample_id in tsm_positive.sample_matrix():
        tsm_positive.set_sample_category(sample_id, 1)
    for sample_id in tsm_unlabeled.sample_matrix():
        category_id = tsm_unlabeled.get_sample_category(sample_id)
        if Categories.get_category_2_id(category_id) == positive_category_id:
            tsm_unlabeled.set_sample_category(sample_id, 1)
        else:
            tsm_unlabeled.set_sample_category(sample_id, -1)

    tsm = tsm_positive.clone()
    tsm.merge(tsm_unlabeled, renewid = False)

    categories = tsm.get_categories()
    print categories

    predict_result = {}
    for sample_id in PS:
        predict_result[sample_id] = 1
    for sample_id in NS:
        predict_result[sample_id] = -1

    clf = BayesianClassifier(categories)
    logging.debug("init_x_in_c_probs() Positive: %d Nagetive: %d ..." % (len(PS), len(NS)))
    clf.init_x_in_c_probs(PS, NS)
    logging.debug("calculate_f_in_c_prob() ...")
    clf.calculate_f_in_c_prob(tsm)

    sample_categories = None
    n = 0
    while True:
        logging.debug("-- %d -- Building the final classifier using P, N, U ..." % (n))
        clf.calculate_x_in_c_prob(tsm_unlabeled)
        sample_categories = clf.result()
        TP, TN, FP, FN = report_iem_result(tsm_P, tsm_U, sample_categories, positive_category_id)

        #if FP == 0:
            #break

        new_predict_result = {}
        for sample_id in sample_categories:
            (likely_category, prob) = sample_categories[sample_id]
            new_predict_result[sample_id] = likely_category
        if new_predict_result == predict_result:
            break
        predict_result = new_predict_result

        logging.debug("-- %d -- Building new NB-C ..." % (n))
        clf.calculate_f_in_c_prob(tsm)
        n += 1

# ---------------- rn_sem() ----------------
def rn_sem(positive_category_id, tsm_positive, tsm_unlabeled, result_dir):
    rn = ReliableNegatives()
    spy_ratio = 0.1
    spy_threshold_ratio = 0.15

    best_log = []
    best_log0 = []

    NS_best = []
    US_best = []
    FP = 0
    FN = 0
    UP = 0
    UN = 0
    FP0 = 0
    FN0 = 0
    UP0 = 0
    UN0 = 0
    common_NS = []
    n = 0
    while n < 1:
        NS, US = rn.S_EM(tsm_positive, tsm_unlabeled, spy_ratio, spy_threshold_ratio, positive_category_id)

        FP, FN, UP, UN = report_sem_result(tsm_positive, tsm_unlabeled, NS, US, positive_category_id)
        best_log.append((FP, FN, UP, UN))

        if n == 0:
            common_NS = NS
        else:
            #if FP == FP0 and FN <= FN0:
                #break
            #common_NS = NS_best
            common_NS = list(set(common_NS).intersection(set(NS)))
            diff_NS = list(set(NS).difference(set(common_NS)))
            US = US + diff_NS
            logging.debug("======== %d common NS (P:%d, U:%d) ========" % (n, len(common_NS), len(US)))
            FP0, FN0, UP0, UN0 = report_sem_result(tsm_positive, tsm_unlabeled, common_NS, US, positive_category_id)
            best_log0.append((FP0, FN0, UP0, UN0))

        NS_best = [i for i in NS]
        US_best = [i for i in US]
        n += 1

    print " \t| FP\t| FN\t| accu\t| UP\t| UN\t|"
    idx = 0
    for (FP, FN, UP, UN) in best_log:
        if FN + FP > 0.0:
            accu = FN / (FN + FP)
        else:
            accu = 0.0
        print "%d\t| %d\t| %d\t| %.3f\t| %d\t| %d\t|" % (idx, FP, FN, accu, UP, UN)
        idx += 1
    print
    print " \t| FP0\t| FN0\t| accu\t| UP0\t| UN0\t|"
    idx = 0
    for (FP, FN, UP, UN) in best_log0:
        if FN + FP > 0.0:
            accu = FN / (FN + FP)
        else:
            accu = 0.0
        print "%d\t| %d\t| %d\t| %.3f\t| %d\t| %d\t|" % (idx, FP, FN, accu, UP, UN)
        idx += 1
    print

    PS = tsm_positive.get_samples_list()

    rocsvm(tsm_positive, tsm_unlabeled, PS, NS_best, US_best, positive_category_id)
    #sem_last(tsm_positive, tsm_unlabeled, PS, NS_best, US_best, positive_category_id)


    #calculate_representative_prototype(tsm_unlabeled, NS_best, US_best)

# ---------------- report_sem_result() ----------------
def report_sem_result(tsm_positive, tsm_unlabeled, NS, US, positive_category_id):
    FN = 0
    FP = 0
    for sample_id in NS:
        category_id = tsm_unlabeled.get_sample_category(sample_id)
        category_1_id = Categories.get_category_1_id(category_id)
        if category_id is None:
            logging.warn("category_id is None in NS. sample_id: %d positive_category_id: %d" % (sample_id, positive_category_id))
        if category_1_id != positive_category_id:
            FN += 1
        else:
            FP += 1

    UP = 0
    UN = 0
    for sample_id in US:
        category_id = tsm_unlabeled.get_sample_category(sample_id)
        if category_id is None:
            logging.warn("category_id is None in US. sample_id: %d positive_category_id: %d" % (sample_id, positive_category_id))
        category_1_id = Categories.get_category_1_id(category_id)
        if category_1_id == positive_category_id:
            UP += 1
        else:
            UN += 1

    print "Total Reliable Negatives: %d" % (FP + FN)
    print "FP: %d FN: %d" % (FP, FN)
    if FN + FP > 0:
        accuracy = FN / (FN +FP)
    else:
        accuracy = 0.0
    print "Accuracy: %.3f" % (accuracy)
    print
    print "Total Unlabeled: %d" % (UP + UN)
    print "UP: %d UN: %d" % (UP, UN)
    print

    return FP, FN, UP, UN


# ---------------- report_iem_result() ----------------
def report_iem_result(tsm_positive, tsm_unlabeled, sample_categories, positive_category_id):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    #report_cnt = 0
    for sample_id in sample_categories:
        (likely_category_id, prob) = sample_categories[sample_id]

        #category_id = tsm.get_sample_category(sample_id)
        category_id = tsm_unlabeled.get_sample_category(sample_id)
        if category_id is None:
            #category_id = tsm_positive.get_sample_category(sample_id)
            continue

        #print "sample_id: %d likely_category_id: %d prob: %.6f" % (sample_id, likely_category_id, prob)

        #if report_cnt < 5:
        #print "report_iem_result() likely_category_id: %d category_id: %d positive_category_id: %d" % (likely_category_id, category_id, positive_category_id)
        #report_cnt += 1

        if Categories.get_category_1_id(category_id) == positive_category_id:
            if likely_category_id == 1:
                TP += 1
            else:
                FP += 1
        else:
            if likely_category_id == -1:
                FN += 1
            else:
                TN += 1
        #logging.debug("sample_id: %d positive_category_id: %d category_id: %d likely_category: %d TP %d FP %d TN %d FN %d" % (sample_id, positive_category_id, category_id, likely_category_id, TP, FP, TN, FN))

    print "\t| True\t| False\t|"
    print "Positive| %d\t| %d\t| %d" % (TP, FP, TP + FP)
    print "Nagetive| %d\t| %d\t| %d" % (TN, FN, TN + FN)
    print
    print "- Positive -"
    positive_accuracy = TP / (TP + TN)
    positive_recall = TP / (TP + FP)
    print "Accuracy: %.3f%%" % (positive_accuracy * 100)
    print "Recall: %.3f%%" % (positive_recall * 100)
    print
    print "- Negative -"
    negative_accuracy = FN / (FN + FP)
    negative_recall = FN / (FN + TN)
    print "Accuracy: %.3f" % (negative_accuracy * 100)
    print "Recall: %.3f" % (negative_recall * 100)
    print

    return TP, TN, FP, FN

