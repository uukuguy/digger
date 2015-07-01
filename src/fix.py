#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
fix.py
'''

import leveldb
import logging
from corpus import Corpus, Samples
from protocal import decode_sample_meta

class Fix():
    def __init__(self, samples):
        self.samples = samples
        self.main_categories = {
            u"安全生产": 0,
            u"电网建设": 1,
            u"经营管理": 2,
            u"供电服务": 3,
            u"业务拓展": 4,
            u"电力改革": 5,
            u"依法治企": 6,
            u"人资管理": 7,
            u"党建作风": 8
            }


    # ---------------- rebuild_categories() ----------------
    def fix_categories(self):
        samples = self.samples
        cat_map = {
                (u"触电伤害", u"") : (u"安全生产", u"触电伤害"),
                (u"其他", u"触电伤害") : (u"安全生产", u"触电伤害"),
                (u"其他", u"意外伤害") : (u"安全生产", u"触电伤害"),
                (u"其他（环境污染）", u"") : (u"安全生产", u"环境保护"),
                (u"其他（破坏电力设施）", u"") : (u"安全生产", u"外力破坏"),
                (u"其他", u"道路破坏") : (u"安全生产", u"外力破坏"),
                (u"其他", u"噪音污染") : (u"安全生产", u"环境保护"),
                (u"其他", u"环境污染") : (u"安全生产", u"环境保护"),
                (u"其他", u"外力破坏") : (u"安全生产", u"外力破坏"),
                (u"其他", u"破坏道路") : (u"安全生产", u"外力破坏"),
                (u"其他（交通安全）", u"") : (u"安全生产", u"交通安全"),
                (u"其他", u"肇事逃逸") : (u"安全生产", u"交通安全"),
                (u"其他", u"酒驾致伤") : (u"安全生产", u"交通安全"),
                (u"其他（树障清除）", u"") : (u"安全生产", u"隐患治理"),
                (u"其他", u"树障清理") : (u"安全生产", u"隐患治理"),
                (u"其他", u"树障清除") : (u"安全生产", u"隐患治理"),
                (u"雾霾", u"") : (u"安全生产", u"隐患治理"),
                (u"安全供电", u"") : (u"安全生产", u"违章作业"),
                (u"智能电网", u"") : (u"电网建设", u"智能电网"),
                (u"新能源并网", u"") : (u"电网建设", u"新能源并网"),
                (u"特高压", u"") : (u"经营管理", u"特高压"),
                (u"阶梯电价", u"") : (u"电力改革", u"电价调整"),
                (u"农网改造", u"") : (u"电力改革", u"农电改制"),
                (u"三集五大", u"") : (u"电力改革", u"三集五大"),
                (u"其他（电农体制改革", u"") : (u"电力改革", u"农电改制"),
                (u"农电改革", u"") : (u"电力改革", u"农电改制"),
                (u"电价调整", u"") : (u"电力改革", u"电价调整"),
                (u"工资福利", u"") : (u"人资管理", u"工资福利"),
                (u"人资劳务", u"") : (u"人资管理", u"人事劳务"),
                (u"人力资源", u"") : (u"人资管理", u""),
                (u"（其他）人资管理",  u"") : (u"人资管理", u""),
                (u"人力资源", u"人事劳务") : (u"人资管理", u"人事劳务"),
                (u"会劳务", u"") : (u"人资管理", u"人事劳务"),
                (u"其他", u"劳动纪律") : (u"人资管理", u"劳动纪律"),
                (u"其他", u"打人致伤") : (u"人资管理", u"劳动纪律"),
                (u"同工同酬", u"") : (u"人资管理", u"同工同酬"),
                (u"作风建设", u"") : (u"党建作风", u""),
                (u"其他", u"作风建设") : (u"党建作风", u""),
                (u"其他", u"舆情宣传") : (u"党建作风", u"新闻宣传"),
                (u"其他", u"新闻宣传") : (u"党建作风", u"新闻宣传"),
                (u"作风建设", u"法律纠纷") : (u"党建作风", u""),
                (u"信访纠纷", u"") : (u"党建作风", u"腐败"),
                (u"腐败", u"") : (u"党建作风", u"腐败"),
                (u"其他", u"公车私用") : (u"党建作风", u"八项规定"),
                (u"腐  败", u"") : (u"党建作风", u"腐败"),
                (u"其他", u"借机敛财") : (u"党建作风", u"腐败"),
                (u"腐败", u"公车购置") : (u"党建作风", u"腐败"),
                (u"农网改造", u"违规收费") : (u"依法治企", u"违规收费"),
                (u"其他", u"强卖") : (u"依法治企", u"违规收费"),
                (u"电费电表", u"违规收费") : (u"依法治企", u"违规收费"),
                (u"其他", u"违规收费") : (u"依法治企", u"违规收费"),
                (u"其他（违规收费）", u"") : (u"依法治企", u"违规收费"),
                (u"其他（乱收费）", u"") : (u"依法治企", u"违规收费"),
                (u"其他", u"违规建房") : (u"依法治企", u"违规建房"),
                (u"其他", u"违规电器") : (u"依法治企", u"违规供电"),
                (u"其他", u"法律纠纷") : (u"依法治企", u"法律纠纷"),
                (u"其他（法律纠纷）", u"") : (u"依法治企", u"法律纠纷"),
                (u"相关利益方", u"") : (u"依法治企", u"审计业务"),
                (u"其他", u"财务审计") : (u"依法治企", u"审计业务"),
                (u"法律纠纷", u"") : (u"依法治企", u"法律纠纷"),
                (u"电动汽车", u"") : (u"业务拓展", u"电动汽车"),
                (u"国际业务", u"") : (u"业务拓展", u"国际业务"),
                (u"风电消纳", u"") : (u"业务拓展", u"产业"),
                (u"供电服务（三指定）", u"") :(u"供电服务", u""),
                (u"智能电表", u"") : (u"供电服务", u"电表"),
                (u"上海停电", u"") : (u"供电服务", u"停电"),
                (u"其他（意外停电）", u"") : (u"供电服务", u"停电"),
                (u"电价", u"") : (u"供电服务", u"电价"),
                (u"其他", u"窃电") : (u"供电服务", u"偷电行为"),
                (u"电费电表", u"") : (u"供电服务", u"电费"),
                (u"电表电费", u"") : (u"供电服务", u"电表"),
                (u"营销服务", u"业务投诉") : (u"供电服务", u"业务投诉"),
                (u"营销服务", u"停电") : (u"供电服务", u"停电"),
                (u"营销服务", u"电费") : (u"供电服务", u"电费"),
                (u"营销服务", u"电价") : (u"供电服务", u"电价"),
                (u"营销服务", u"电表") : (u"供电服务", u"电表"),
            }

        #bad_samples = []

        rowidx = 0
        for i in samples.db_content.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                continue
            (sample_id, category, date, title, key, url, msgext) = decode_sample_meta(i[1])
            (version, content, (cat1, cat2, cat3)) = msgext
            #try:
                #(version, content, (cat1, cat2, cat3)) = msgext
            #except ValueError:
                #bad_samples.append(sample_id)
            #cat1 = cat1.decode('utf-8')
            #cat2 = cat2.decode('utf-8')
            #cat3 = cat3.decode('utf-8')
            #if cat1 == u"农电改革":
                #logging.debug("<%s:%s:%s:>" % (cat1, cat2, cat3))
                #if (cat1, cat2) in cat_map:
                    #logging.debug("Found <%s:%s::> in cat_map" % (cat1, cat2))
                #else:
                    #logging.debug("Not found <%s:%s::> in cat_map" % (cat1, cat2))
                    #print cat2.__class__, (cat1, cat2) == (cat1, u""), (cat1, cat2) == (cat1, u"")


            new_cat3 = cat3
            #if cat2 == u"":
                #print "cat2 == NULL <%s:%s:%s:>" % (cat1, cat2, cat3)
            if (cat1, cat2) in cat_map:
                new_cat1, new_cat2 = cat_map[(cat1, cat2)]
                str_sample_meta = (sample_id, category, date, title, key, url, (version, content, (new_cat1, new_cat2, new_cat3)))
                samples.db_content.Put(str(sample_id), msgpack.dumps(str_sample_meta))
                logging.debug("<%s:%s:%s:> -> <%s:%s:%s:>" % (cat1, cat2, cat3, new_cat1, new_cat2, new_cat3))

            rowidx += 1

    # ---------------- refresh_content() ----------------
    def refresh_content(self):
        db_content = self.samples.db_content
        urls = []
        rowidx = 0
        for i in db_content.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                continue
            (sample_id, category, date, title, key, url, msgext) = decode_sample_meta(i[1])
            (version, content, (cat1, cat2, cat3)) = msgext


            if content is None:
                logging.debug("content is None: sample_id %d" % (sample_id))
                urls.append((sample_id, category, date, title, key, url, cat1, cat2, cat3))
            elif len(content) == 0 :
                logging.debug("len(conntent) == 0: sample_id %d" % (sample_id))
                urls.append((sample_id, category, date, title, key, url, cat1, cat2, cat3))

            if rowidx % 100 == 0:
                logging.debug("refresh content - %d" % (rowidx))
            rowidx += 1

        for (sample_id, category, date, title, key, url, cat1, cat2, cat3) in urls:
            logging.debug("--------------------------------")
            logging.debug("sample_id: %d url:%s" % (sample_id, url))
            try:
                rsp = requests.get(url)
                if rsp.ok:
                    #filename = "no_%d.html" % sample_id
                    #print rsp.text.encode('utf-8')
                    #f = open(filename, "wb+")
                    #f.write(rsp.text.encode('utf-8'))
                    #f.close()
                    content = rsp.text
                    version = "1"
                    msgext = (version, content, (cat1, cat2, cat3))

                    sample_data = (sample_id, category, date, title, key, url, msgext)
                    rowstr = msgpack.dumps(sample_data)
                    db_content.Put(str(sample_id), rowstr)
                else:
                    version = "1"
                    msgext = (version, "", (cat1, cat2, cat3))
                    sample_data = (sample_id, category, date, title, key, url, msgext)
                    rowstr = msgpack.dumps(sample_data)
                    db_content.Put(str(sample_id), rowstr)
                    logging.warn("Get page failed. status: %d sample_id: %d url: %s" % (rsp.status_code, sample_id, url))
            except:

                version = "1"
                msgext = (version, None, (cat1, cat2, cat3))
                sample_data = (sample_id, category, date, title, key, url, msgext)
                rowstr = msgpack.dumps(sample_data)
                db_content.Put(str(sample_id), rowstr)
                logging.warn("Connection failed. sample_id: %d url: %s" % (sample_id, url))




    # ---------------- purge() ----------------
    # 删除db_content中所有content == None的记录。
    # 通常是多次连接不上，无法获得实际文本内容的记录。
    def purge(self):
        samples = self.samples
        db_content = samples.db_content

        none_samples, empty_samples, _ = samples.get_bad_samples()
        purged_samples = [ sample_id for (sample_id, url) in none_samples]

        logging.debug("Purgging %d samples...." % (len(purged_samples)))
        total_samples = samples.get_total_samples()

        for sample_id in purged_samples:
            db_content.Delete(str(sample_id))
            logging.debug("Purge None content sample %d" % (sample_id))
        total_samples -= len(purged_samples)

        for (sample_id, url) in empty_samples:
            db_content.Delete(str(sample_id))
            logging.debug("Purge empty content sample %d" % (sample_id))
        total_samples -= len(empty_samples)

        logging.debug("Purge Done. Remaining %d samples." % (total_samples))


        invalid_class_samples = []
        invalid_categories = {}
        rowidx = 0
        for i in db_content.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                continue
            (sample_id, category, date, title, key, url, msgext) = decode_sample_meta(i[1])
            (version, content, (cat1, cat2, cat3)) = msgext

            if not cat1 in self.main_categories:
                invalid_class_samples.append(sample_id)
                if (cat1, cat2) in invalid_categories:
                    invalid_categories[(cat1, cat2)] += 1
                else:
                    invalid_categories[(cat1, cat2)] = 1

        for (cat1, cat2) in invalid_categories:
            logging.debug("<I> <%s:%s::> %d" % (cat1, cat2, invalid_categories[(cat1, cat2)]))

        logging.debug("Total invalid class samples %d in %d categories" % (len(invalid_class_samples), len(invalid_categories)) )


        for sample_id in invalid_class_samples:
            db_content.Delete(str(sample_id))
        logging.debug("Deleted %d invalid class samples." % (len(invalid_class_samples)))

