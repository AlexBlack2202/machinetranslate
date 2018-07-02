#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup, SoupStrainer,Comment
import requests
import codecs
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support import expected_conditions as EC

lst_data =[]

chromeOptions = webdriver.ChromeOptions()
prefs = {"profile.managed_default_content_settings.images":2}
chromeOptions.add_experimental_option("prefs",prefs)
#driver = webdriver.PhantomJS(service_args=['--ignore-ssl-errors=true', '--ssl-protocol=TLSv1'],executable_path='C:\phantomjs\phantomjs.exe')
driver = webdriver.Chrome(chrome_options=chromeOptions,service_args=['--ignore-ssl-errors=true', '--ssl-protocol=TLSv1'],executable_path='C:\phantomjs\chromedriver.exe')
def fun_extract_link(page_url):
    driver.get(page_url)

    #driver.save_screenshot('lotte_1.png');

    timeout = 10
    print("waiting %s seconds..." % timeout)
    wait = WebDriverWait(driver, timeout)
    list_product = wait.until(EC.visibility_of(driver.find_element_by_class_name("products-list")))

    driver.save_screenshot('images/'+page_url+'lotte_1.png');
    print("done waiting. Response: " + page_url)

    int_count = 0

    list_product = driver.find_element_by_class_name('main')

    if list_product is None:
        print("item is None")
        return int_count

    for link in list_product.find_elements_by_tag_name('a'):
        link_url = link.get_attribute("href")
        
        try:
            if link_url is not None and len(link_url)>5 and link_url.find('https:')>-1 :
                lst_data.append(str(link_url))
                int_count = int_count+1
        except Exception  as ex:
            print(ex)
            pass

    # with open('tiki.txt','w+') as handler:
    #     handler.writelines(('\n'.join(lst_data).encode('utf-8').decode('utf-8')))

    return int_count


url="https://www.lotte.vn/category/170/suc-khoe-lam-dep?page="

for item in range(1,5000):
    cur_url = url+str(item)
    print(cur_url)
    if fun_extract_link(cur_url)  <=0:
        break

file = codecs.open("lotte_url.txt", "w+", "utf-8")
for item in lst_data:
    file.write(item.strip())
    file.write('\r\n')
file.close()