这份数据是2018年年中抓取的网易严选商品属性和商品评价数据。

commAttri.xlsx：商品属性
	itemId：商品ID
	minTime：商品评价的最早生成日期
	Ncmts：抓取数据时刻，商品的评价总量
	prodName：商品名称
	category：商品类别
	itemTagList：商品标签
	promTag：抓取数据时刻，商品的促销标签
	productPlace：商品产品标签
	price：抓取数据时刻，商品的标价

csv-cmt文件夹中每一个文件存储一个商品的全部评价数据。文件名前缀为商品ID，以此ID至商品属性表commAttri.xlsx的itemId列找到对应行，即可获取此商品的属性信息。文件中每一行为一条商品评论：
	itemId：商品ID
	star：商品评分
	content：评价内容
	createTime：评价生成时间
