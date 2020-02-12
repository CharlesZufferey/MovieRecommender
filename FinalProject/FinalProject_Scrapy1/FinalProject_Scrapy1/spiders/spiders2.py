# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:15:53 2020

@author: Charles
"""

import scrapy
from scrapy.http import Request
import re
#%%

class FinalprojectScrapy1Item(scrapy.Item):
    image_urls = scrapy.Field()
    images = scrapy.Field()
    sku = scrapy.Field()
    #producttype = scrapy.Field()
    
class Images(scrapy.Spider):
    #define the name of the spider which we will call from CLI
    name = 'scrape_images2'
    #set the range of urls which can be scraped over
    allowed_domain = ['https://www.wayfair.com']#/castleton-home-fairy-new-age-reflexology-feet-figurine-ccoo1930.html']
    start_urls = ['https://www.wayfair.com/furniture/pdp/hashtag-home-clopton-215-armchair-w002769569.html?piid=1619703395']

#%%
    
    def parse(self, response):                                                   

        
        #url = response.css('a[class*="Link Link--primary"]').xpath('@href').getall()
        url = response.xpath('//img[contains(@src,"h800")]/@src').getall()
           #print(url)
        for link in url:
            # sku = response.css('span[class*=Breadcrumbs-item]::text').re(r': (\w+)')    
            # item=FinalprojectScrapy1Item()
            # item['image_urls'] = url
            # item['sku'] = sku
            # yield item
            print (link)