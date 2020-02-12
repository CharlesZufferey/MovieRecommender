# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:57:32 2020

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
    name = 'scrape_images'
    #set the range of urls which can be scraped over
    allowed_domain = ['https://www.wayfair.com']#/castleton-home-fairy-new-age-reflexology-feet-figurine-ccoo1930.html']
    #start_urls = ['https://www.wayfair.co.uk/furniture/pdp/borough-wharf-prater-3-tier-bookcase-wlfg2014.html']
    #start_urls = ['https://www.wayfair.co.uk/furniture/sb0/console-tables-c225559.html?curpage=16']
    #start_urls = ['https://www.wayfair.co.uk/furniture/sb0/ottomans-pouffes-c1802512.html?curpage=16']
    #start_urls = ['https://www.wayfair.co.uk/furniture/sb0/armchairs-c493383.html?curpage=18']
    #start_urls = ['https://www.wayfair.co.uk/furniture/sb0/chest-of-drawers-c224865.html?curpage=19']
    #start_urls = ['https://www.wayfair.co.uk/home-decor/sb0/wall-mirrors-c1802010.html?curpage=21']
    start_urls = ['https://www.wayfair.com/keyword.php?keyword=grey+chair']
    #url = ''
#%%
    
    def parse(self, response):                                                   
#        #a[href*=image]::attr(href)
#        url = response.css('a[href*="furniture-c1852173"]').xpath('@href').get() #a[contains(@href,"home-decor-c225067")]
#        print(url)
#        yield scrapy.Request(url,self.parse_page1)
#        
#    def parse_page1(self, response):
#        url = response.xpath('//a[contains(@href,"bookcases")]/@href').get()
#        #print(url)
#        yield scrapy.Request(url,self.parse_cat)
        
    #def parse_cat(self, response):    
        #bookcases page 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
        #console tables page 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
        #ottomans page 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
        #armchairs page 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
        #Chest of drawers page 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
        #mirrors page 2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21
        #kitchen islands 2,3,4,5,6,7,8
        
        url = response.css('a[target*="_blank"]').xpath('@href').getall()
        for link in url:
            yield scrapy.Request(link, self.parse_page2)
#
            #print (link)

#        
    def parse_page2(self,response):
        
        url = response.xpath('//img[contains(@src,"h800")]/@src').getall()
        #url1 = response.css('ul[class*="pl-CarouselContainer pl-MultiCarousel-slider ProductDetailImageCarouselVariantB-carousel"] li div div div img').xpath('@src').extract()
        #sku = response.xpath('//div[@id="bd"]/span[contains(@class,"SKU:")]/text()').extract_first()
        sku = response.css('span[class*=Breadcrumbs-item]::text').re(r': (\w+)')    

        
        item=FinalprojectScrapy1Item()
        item['image_urls'] = url
        item['sku'] = sku
        yield item
        

        
        
        
        
        
        
        
        
        