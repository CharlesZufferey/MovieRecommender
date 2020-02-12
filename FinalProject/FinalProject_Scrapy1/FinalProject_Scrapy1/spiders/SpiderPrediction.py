# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:44:14 2020

@author: Charles
"""
"""
Correct command line: scrapy runspider -o output.csv -L WARNING SpiderPrediction.py -a start_url="https://www.wayfair.com/keyword.php?keyword=wooden+table&command=dosearch&new_keyword_search=true&class_id="
"""
#%%
import scrapy
from scrapy.http import Request
import re
#%% Class to scrape images
class FinalprojectScrapy1Item(scrapy.Item):
    image_urls = scrapy.Field()
    images = scrapy.Field()
    sku = scrapy.Field()
    
class Images(scrapy.Spider):
    #define the name of the spider which we will call from CLI
    name = 'scrape_test2'
    #set the range of urls which can be scraped over
    allowed_domain = ['https://www.wayfair.com']
    #start_urls = ['https://www.wayfair.com/keyword.php?keyword=wooden+table&command=dosearch&new_keyword_search=true&class_id=']
    def __init__(self, *args, **kwargs): 
        super(Images, self).__init__(*args, **kwargs) 
        self.start_urls = [kwargs.get('start_url')] 
    
    def parse(self, response):            
        """Initial Function - looking at all products on page and extracting link"""                                       
        url = response.css('a[target*="_blank"]').xpath('@href').getall()
        for link in url:
            yield scrapy.Request(link, self.parse_page2)

    def parse_page2(self,response):
        """Following link and downloading image into folder"""
        url = response.xpath('//img[contains(@src,"h800")]/@src').getall()
        sku = response.css('span[class*=Breadcrumbs-item]::text').re(r': (\w+)')    
        item=FinalprojectScrapy1Item()
        item['image_urls'] = url
        item['sku'] = sku
        yield item

#%%
