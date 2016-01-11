#mongodb 查询脚本

脚本根据指定时间段进行查询并生成csv格式(utf-8)

配置文件说明

	"host": mongodb ip
	"port": 端口
	"begin_date": 开始时间
	"end_date": 结束时间
	"fieldnames": 生成的结果字段keyword,publish_time,title,text,type,url, 可以根据需要保留字段,以逗号分隔
	"search_collection":"all",舆情数据集合,默认为all,每个结果集生成应该csv文件, 可以单独指定查询特定mongodb数据集合
	

目前mongodb数据集合有	

baidu_site_search 百度网页搜索

baidu_tieba  百度贴吧

common 指定的4000个站点采集

hotpoint_baidu  百度热点

hotword_baidu  百度热词

news_360  360搜索

news_baidu  百度新闻

news_sougou  搜狗新闻

sougou_wenxin 搜狗微信

weibo_qq 腾讯微博

weibo_sina 新浪微博

weibo_renmin 人民网微博

weibo_xinhua 新华网微博


##例子:

###`python main.c -c config.json`
	
	{
	  "host":"139.196.189.136",
	  "port":27017,
	  "begin_date":"2016-01-01",
	  "end_date":"2016-01-02",
	  "fieldnames":"publish_time,text",
	  "search_collection":"weibo_qq"
	}

只生成weibo_qq的数据,包含了发布时间和内容

