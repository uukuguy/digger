#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
'''

import os
from os import path
from bidict import bidict
import logging
from logger import Logger
import leveldb
import jieba
#from pyltp import Segmentor
from utils import is_chinese_word

class SegmentMethod():
    JIEBA = 0
    NGRAM = 1
    #LTP = 2

class Vocabulary:

    # ---------------- __init__() ----------------
    def __init__(self, vocabulary_dir):
        #self.maxid = 0
        #self.terms = bidict()
        self.terms_by_text = {}
        self.terms_by_id = {}
        self.root_dir = vocabulary_dir

        self.batch_vocabulary = None
        #self.db = leveldb.LevelDB(self.root_dir)
        self.db = None
        #self.load()

        self.segger = None

    # ---------------- clear() ----------------
    def clear(self):
        #self.terms = bidict()
        self.terms_by_text = {}
        self.terms_by_id = {}

    # ---------------- get_terms_count() ----------------
    def get_terms_count(self):
        #return len(self.terms)
        return len(self.terms_by_text)

    # ---------------- add_term() ----------------
    def add_term(self, term_text):
        #term_id = self.maxid
        #self.maxid += 1
        #return term_id

        #term_id = self.terms.setdefault(term_text, len(self.terms))
        if not term_text in self.terms_by_text:
            term_id = self.get_terms_count()
            self.terms_by_text[term_text] = term_id
            self.terms_by_id[term_id] = term_text
            if self.batch_vocabulary is None:
                self.batch_vocabulary = leveldb.WriteBatch()
            self.batch_vocabulary.Put(str(term_id), term_text.encode('utf-8'))
        else:
            term_id = self.get_term_id(term_text)

        return term_id

    # ---------------- get_term_id() ----------------
    def get_term_id(self, term_text):
        #term_id = self.terms[term_text]
        term_id = self.terms_by_text[term_text]
        return term_id

    # ---------------- get_term_text() ----------------
    def get_term_text(self, term_id):
        #term_text = self.terms[:term_id]
        if term_id in self.terms_by_id:
            term_text = self.terms_by_id[term_id]
        else:
            if self.db is None:
                self.db = leveldb.LevelDB(self.root_dir)
            term_text = self.db.Get(str(term_id))
            if term_text is None:
                term_text = u""
            else:
                term_text = term_text.decode('utf-8')
            self.db = None
        return term_text

    # ---------------- save() ----------------
    def save(self):
        #batch_vocabulary = leveldb.WriteBatch()

        ##for term_text in self.terms:
            ##term_id = self.terms[term_text]
        #for term_id in self.terms_by_id:
            #term_text = self.terms_by_id[term_id]
            #batch_vocabulary.Put(str(term_id), term_text.encode('utf-8'))

        if self.db is None:
            self.db = leveldb.LevelDB(self.root_dir)
        if not self.batch_vocabulary is None:
            self.db.Write(self.batch_vocabulary, sync=True)
            self.batch_vocabulary = None
        self.db = None

    # ---------------- load() ----------------
    def load(self):
        if self.db is None:
            self.db = leveldb.LevelDB(self.root_dir)

        rowidx = 0
        for i in self.db.RangeIter():
            row_id = i[0]
            if row_id[0:2] == "__":
                continue
            term_id = int(row_id)
            #logging.debug(Logger.debug("%s" % (str(i[1].__class__))))
            term_text = i[1].decode('utf-8')

            #self.terms.setdefault(term_text, term_id)
            self.terms_by_id[term_id] = term_text
            self.terms_by_text[term_text] = term_id

            if rowidx % 10000 == 0:
                logging.debug(Logger.debug("%d %d:%s" % (rowidx, term_id, term_text)))
            rowidx += 1

        self.db = None

    # ---------------- seg_content() ----------------
    def seg_content(self, content):
        term_map = self.add_text(content)
        sample_terms = 0
        for term_id in term_map:
            sample_terms += term_map[term_id]

        return sample_terms, term_map

    # ---------------- add_text() ----------
    def add_text(self, content, segment_method = SegmentMethod.JIEBA):
        if segment_method == SegmentMethod.JIEBA:
            return self.add_text_jieba(content)
        elif segment_method == SegmentMethod.NGRAM:
            return self.add_text_ngram(content)
        #elif segment_method == SegmentMethod.LTP:
            #return self.add_text_ltp(content)
        else:
            raise ValueError, ("Invalid SegmentMethod.")

    # ---------------- add_text_jieba() ----------------
    def add_text_jieba(self, content):
        term_map = {}
        #jieba.enable_parallel(4)
        tokens = jieba.cut(content)
        for fet in tokens:
            u0 = fet[0]
            if not is_chinese_word(u0) :
                continue
            if len(fet) < 2:
                continue

            term_id = self.add_term(fet)

            if term_id in term_map:
                term_map[term_id] += 1
            else:
                term_map[term_id] = 1

        return term_map

    # ---------------- add_text_ngram() ----------------
    def add_text_ngram(self, content):
        term_map = {}
        len_c = len(content)
        for i in range(0, len_c - self.N + 1):
            if i > self.MAX_FETS:
                break
            fet = content[i: i + self.N]
            term_id = self.add_term(fet)

            if term_id in term_map:
                term_map[term_id] += 1
            else:
                term_map[term_id] = 1

        return term_map


    # ---------------- add_text_ltp() ----------------
    #def add_text_ltp(self, content):
        #term_map = {}

        #if self.segger is None:
            #self.segger = Segmentor()
            #self.segger.load('/home/jwsu/apps/ltp-3.3.0/ltp_data/cws.model')
        #tokens = self.segger.segment(content.encode('utf8'))
        #for fet in tokens:
            #u0 = fet.decode('utf8')
            #if not is_chinese_word(u0) :
                #continue
            #if len(fet) < 2:
                #continue

            #term_id = self.add_term(fet.decode('utf8'))

            #if term_id in term_map:
                #term_map[term_id] += 1
            #else:
                #term_map[term_id] = 1

        #return term_map


