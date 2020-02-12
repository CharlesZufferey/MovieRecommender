# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 09:54:12 2020

@author: Charles
"""

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
    name = 'scrape_images2'
    #set the range of urls which can be scraped over
    allowed_domain = ['https://www.wayfair.co.uk']#/castleton-home-fairy-new-age-reflexology-feet-figurine-ccoo1930.html']
    start_urls = ['https://www.wayfair.co.uk/furniture/pdp/borough-wharf-prater-3-tier-bookcase-wlfg2014.html']
    #start_urls = [https://www.wayfair.co.uk/furniture/sb0/bookcases-c493393.html?curpage=9]
    #url = ''
#%%
    
    def parse(self, response):
        url = response.xpath('//ul[contains(@class,"pl-CarouselContainer pl-MultiCarousel-slider ProductDetailImageCarouselVariantB-carousel")]').get()
        
        #url = response.css('ul[class*="pl-CarouselContainer pl-MultiCarousel-slider ProductDetailImageCarouselVariantB-carousel"]').xpath('@src').get() #a[contains(@href,"home-decor-c225067")]
        print(url)
#        url = response.xpath('//a[contains(@href,"bookcases")]/@href').get()
#         
#%%

        
        
        
        
        
        
        
        
        
        