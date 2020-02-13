# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:50:24 2020

@author: Charles
"""

import os  
from selenium import webdriver  
from selenium.webdriver.common.keys import Keys  
from selenium.webdriver.chrome.options import Options  
#%%
chrome_options = Options()  
chrome_options.add_argument("--headless")  
#chrome_options.binary_location = '/Applications/Google Chrome   Canary.app/Contents/MacOS/Google Chrome Canary'`    
#%%
driver = webdriver.Chrome(executable_path=os.path.abspath("chromedriver"),   chrome_options=chrome_options)  
driver.get("http://www.duo.com")
#%%
magnifying_glass = driver.find_element_by_id("js-open-icon")  
if magnifying_glass.is_displayed():  
  magnifying_glass.click()  
else:  
  menu_button = driver.find_element_by_css_selector(".menu-trigger.local")  
  menu_button.click()`  

search_field = driver.find_element_by_id("site-search")  
search_field.clear()  
search_field.send_keys("Olabode")  
search_field.send_keys(Keys.RETURN)  
assert "Looking Back at Android Security in 2016" in driver.page_source   driver.close()`  
#%%
import time
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()  # Optional argument, if not specified will search path.
driver.get('https://www.wayfair.co.uk/furniture/pdp/natur-pur-4-drawer-chest-of-drawers-hzel1060.html');
time.sleep(5) # Let the user actually see something!
#button = driver.find_element_by_xpath("//button[contains(@class,'CarouselButton')]")
element = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, "//div/img/[contains(@src,'7700596')]")))
driver.execute_script("arguments[0].click();", element)
#search_box.send_keys('beds')
#search_box.submit()
#button.click()
#time.sleep(5) # Let the user actually see something!
#driver.quit()
#%%
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(chrome_options=options)#, executable_path=r'C:\Utility\BrowserDrivers\chromedriver.exe', service_args=["--log-path=./Logs/DubiousDan.log"])
print ("Headless Chrome Initialized")
params = {'behavior': 'allow', 'downloadPath': r'C:\Users\Charles\Downloads'}
driver.execute_cdp_cmd('Page.setDownloadBehavior', params)
driver.get("https://www.wayfair.co.uk/furniture/pdp/natur-pur-4-drawer-chest-of-drawers-hzel1060.html")
driver.execute_script("scroll(0, 250)"); 
WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH("//button[contains(@class,'CarouselButton')]")))).click()
print ("Download button clicked")
#driver.quit()