#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup, SoupStrainer,Comment
import requests
import codecs

lst_data =[]

def fun_extract_link(page_url):
    page = requests.get(page_url)    
    data = page.text
    # f = codecs.open("data/tiki.vn.html", "r", "utf-8")
    # data = f.read()
    # data = data.encode('utf-8').decode('utf-8')
    def getText(parent):
        return ''.join(parent.find_all(text=True, recursive=False)).strip()

    soup = BeautifulSoup(data,"html.parser")

    comments = soup.findAll(text=lambda text:isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    

    list_product = soup.find('div', {'class': 'product-box-list'})

    int_count = 0

    if list_product is None:
        print("item is None")
        return int_count

    for link in list_product.find_all('a'):
        link_url = link.get('href')
        
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


url="https://tiki.vn/lam-dep-suc-khoe/c1520?src=tree&page="

for item in range(1,500):
    cur_url = url+str(item)
    print(cur_url)
    if fun_extract_link(cur_url)  <=0:
        break

file = codecs.open("tiki_url.txt", "w+", "utf-8")
for item in lst_data:
    file.write(item.strip())
    file.write('\r\n')
file.close()