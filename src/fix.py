#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
fix.py
'''

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



    # ---------------- get_bad_samples() ----------------
    def get_bad_samples(self):
        samples = self.samples

        none_samples = []
        empty_samples = []
        normal_samples = []
        rowidx = 0
        for i in samples.db_content.RangeIter():
            row_id = i[0]
            if row_id.startswith("__"):
                continue
            (sample_id, category, date, title, key, url, msgext) = decode_sample_meta(i[1])
            (version, content, (cat1, cat2, cat3)) = msgext

            if content is None:
                none_samples.append((sample_id, url))
            elif len(content) == 0:
                empty_samples.append((sample_id, url))
            else:
                normal_samples.append((sample_id, url))

            rowidx += 1

        logging.debug("Get %d bad samples. None: %d Empty: %d Normal: %d" % (len(none_samples) + len(empty_samples) +len(normal_samples), len(none_samples), len(empty_samples), len(normal_samples)))

        return none_samples, empty_samples, normal_samples


    # ---------------- purge() ----------------
    # 删除db_content中所有content == None的记录。
    # 通常是多次连接不上，无法获得实际文本内容的记录。
    def purge(self):
        samples = self.samples
        db_content = samples.db_content

        none_samples, empty_samples, _ = self.get_bad_samples()
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

