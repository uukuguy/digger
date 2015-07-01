
负面舆情历史数据 (po2014.corpus/samples/no2014)
原始记录数：5761 其中坏数据(URL不可访问) 177  
完整数据：558条 79M


dead samples:
5884
5909
6078
6405 
6453
6460


# ---------
./digger.py import --corpus_dir po2014.corpus --samples_name 2014_neg_1Q --xls_file ./data/po2014_neg_1Q.xls

./digger.py rebuild --corpus_dir po2014.corpus --samples_name 2014_neg_1Q

./digger.py import --corpus_dir po2014.corpus --samples_name po201405 --xls_file ./data/po201405_full.xls

./digger.py rebuild --corpus_dir po2014.corpus --samples_name po201405

./digger.py learn --corpus_dir po2014.corpus --positive_name 2014_neg_1Q --unlabeled_name po201405 --model_file po2014.model

# ---------
./digger.py refresh --corpus_dir po2014.corpus --samples_name no2014
./digger.py purge --corpus_dir po2014.corpus --samples_name no2014



# ---------
./digger.py query_categories --corpus_dir po2014.corpus --samples_name no2013 --xls_file categories.xls 

./digger.py query_keywords --corpus_dir po2014.corpus --samples_name no2013 --result_dir ./result

./digger.py query_sample --corpus_dir po2014.corpus --samples_name no2014 --sample_id 444

./digger.py show --corpus_dir po2014.corpus --samples_name no2013


# ---------
./digger.py export_samples --corpus_dir po2014.corpus --samples_name no2013 --xls_file no2013_samples.xls 

./digger.py export_urls --corpus_dir po2014.corpus --samples_name no2013 --xls_file no2013_urls.xls 

# ---------
./digger.py sne --corpus_dir po2014.corpus --samples_name no2013 

# ---------

./digger.py train --corpus_dir po2014.corpus --samples_name no2013 --model_name no2013
./digger.py predict --corpus_dir po2014.corpus --samples_name no2013 --model_name no2013

# ---------
./digger.py import_samples --corpus_dir po2014.corpus --samples_name no2014_cleaned --xls_file ~/apps/digger/data/po2014_cleaned.xls

./digger.py rebuild --corpus_dir po2014.corpus --samples_name no2014_cleaned 

./digger.py purge --corpus_dir po2014.corpus --samples_name no2014_cleaned 



Dependencies:

pip install xlrd xlwt xlsxwriter
pip install docopt
pip install bidict shutil leveldb msgpack-python requests
pip install sklearn-learn sklearn-pandas statsmodels patsy seaborn 
pip install jieba
pip install textract beautifulsoup4  antiword python-docx python-pptx 

pip install bokeh scaly
pip install flask tornado

pip install json-c 


