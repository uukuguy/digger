#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
reliable_negatives.py - 可信负例样本抽取算法
'''

from __future__ import division
import logging
from utils import sorted_dict
from term_sample_matrix import TermSampleMatrix
from categories import Categories

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

    def __calculate_f_in_c_base(self, tsm, c):
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
            sample_prob *= self.get_f_in_c_prob(c, term_id)
            #logging.debug("base - %d %.3f" % (term_id, sample_prob))
        return sample_prob

    # ---------------- calculate_x_in_c_prob() ----------
    # self.Pr_f_in_c -> self.Pr_x_in_c
    # 更新所有未标记样本的后验概率Pr[c1|di]
    def calculate_x_in_c_prob(self, tsm, positive_sample_max_id):
        categories = self.categories
        for sample_id in tsm.sample_matrix():
            # 所有正例样本的分类概率保持不变
            #if sample_id < positive_sample_max_id:
                #continue

            sample_prob_sum = 0.0
            sample_probs = []
            for category_id in categories:
                Pr_c = self.get_c_prob(category_id)
                sample_prob_base = self.__calculate_x_in_c_base(tsm, category_id, sample_id)
                sample_prob = Pr_c * sample_prob_base
                sample_probs.append((category_id, sample_prob))
                sample_prob_sum += sample_prob
                #logging.debug("Pr_c: %.3f sample_prob_base: %.9f" % (Pr_c, sample_prob_base))
            if sample_prob_sum > 0.0:
                for (category_id, sample_prob) in sample_probs:
                    prob = sample_prob / sample_prob_sum
                    self.set_x_in_c_prob(category_id, sample_id, prob)
            else:
                for category_id in categories:
                    self.set_x_in_c_prob(category_id, sample_id, 0.0)


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

        return sample_categories


class ReliableNegatives():

    # ---------------- __init__() ----------------
    def __init__(self, tsm_positive, tsm_unlabeled, positive_category_id):
        self.tsm_positive = tsm_positive
        self.tsm_unlabeled = tsm_unlabeled
        self.positive_category_id = positive_category_id

    def __init_samples(self):
        tsm_positive = self.tsm_positive.clone()
        tsm_unlabeled = self.tsm_unlabeled.clone()

        positive_sample_max_id = tsm_positive.sample_max_id
        total_unlabeled_samples = tsm_unlabeled.get_total_samples()
        for sample_id in tsm_positive.sample_matrix():
            tsm_positive.set_sample_category(sample_id, 1)
        for sample_id in tsm_unlabeled.sample_matrix():
            tsm_unlabeled.set_sample_category(sample_id, -1)

        tsm = tsm_positive.clone()
        #tsm.merge(tsm_unlabeled, renewid = True)
        tsm.merge(tsm_unlabeled, renewid = False)
        tsm.rebuild_categories()

        return tsm, tsm_positive, tsm_unlabeled, positive_sample_max_id


    # ---------------- __init__() ----------------
    def I_EM(self):
        tsm, tsm_positive, tsm_unlabeled, positive_sample_max_id = self.__init_samples()
        logging.debug("positive_sample_max_id: %d" % (positive_sample_max_id))
        categories = tsm.get_categories()
        print categories

        # Build initial Naive Bayesian Classifier NB-C
        clf = BayesianClassifier(categories)
        predict_result = {}
        for sample_id in tsm.sample_matrix():
            category_id = tsm.get_sample_category(sample_id)
            if category_id == 1:
                clf.set_x_in_c_prob(1, sample_id, 1.0)
                clf.set_x_in_c_prob(-1, sample_id, 0.0)
                predict_result[sample_id] = 1
            else:
                clf.set_x_in_c_prob(1, sample_id, 0.0)
                clf.set_x_in_c_prob(-1, sample_id, 1.0)
                predict_result[sample_id] = -1

            #if sample_id < positive_sample_max_id:
                #clf.set_x_in_c_prob(1, sample_id, 1.0)
                #clf.set_x_in_c_prob(-1, sample_id, 0.0)
            #else:
                #clf.set_x_in_c_prob(1, sample_id, 0.0)
                #clf.set_x_in_c_prob(-1, sample_id, 1.0)
        #print clf.Pr_x_in_c

        clf.calculate_f_in_c_prob(tsm)

        # Classify tsm using current NB-C
        #while True:
            #clf.calculate_x_in_c_prob(tsm, positive_sample_max_id)

        sample_categories = None
        n = 0
        while True:
            logging.debug("-- %d -- Predicting ..." % (n))
            clf.calculate_x_in_c_prob(tsm_unlabeled, positive_sample_max_id)
            sample_categories = clf.result()
            TP, TN, FP, FN = report_result(self.tsm_positive, self.tsm_unlabeled, sample_categories, self.positive_category_id)

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



def rn_iem(positive_category_id, tsm_positive, tsm_unlabeled, result_dir):
    rn = ReliableNegatives(tsm_positive, tsm_unlabeled, positive_category_id)
    tsm, sample_categories = rn.I_EM()

    #report_result(tsm_positive, tsm_unlabeled, sample_categories, positive_category_id)


def report_result(tsm_positive, tsm_unlabeled, sample_categories, positive_category_id):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    positive_sample_max_id = tsm_positive.sample_max_id
    logging.debug("positive_sample_max_id: %d" % (positive_sample_max_id))
    for sample_id in sample_categories:
        (likely_category_id, prob) = sample_categories[sample_id]

        #category_id = tsm.get_sample_category(sample_id)
        category_id = tsm_unlabeled.get_sample_category(sample_id)
        if category_id is None:
            #category_id = tsm_positive.get_sample_category(sample_id)
            continue


        #if sample_id < positive_sample_max_id:
            #category_id = tsm_positive.get_sample_category(sample_id)
        #else:
            #category_id = tsm_unlabeled.get_sample_category(sample_id - positive_sample_max_id)
        #if category_id == 1:

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
