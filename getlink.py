#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup, SoupStrainer,Comment
import requests
import codecs


url = "https://tiki.vn/"

# page = requests.get(url)    
# data = page.text
f = codecs.open("data/tiki.vn.html", "r", "utf-8")
data = f.read()
data = data.encode('utf-8').decode('utf-8')
def getText(parent):
    return ''.join(parent.find_all(text=True, recursive=False)).strip()

soup = BeautifulSoup(data,"html.parser")

comments = soup.findAll(text=lambda text:isinstance(text, Comment))
for comment in comments:
    comment.extract()
lst_data =[]
for link in soup.find_all('a'):
    link_url = link.get('href')
    text =link.get('title')

    
    
    if text is None or len(text)<1:
        text =link.get('data-title')

    li = link.find('p', {'class': 'price-sale'})
    
    
        
    try:
        if li is not None and text is not None and link_url is not None and len(link_url)>5 and len(text)>0 and link_url.find('.html')>1 and link_url.find('https:')>-1 :
            lst_data.append(str(link_url)+"\t"+str(text.strip())+"\t"+getText(li))
    except Exception  as ex:
        print(ex)
        pass

# with open('tiki.txt','w+') as handler:
#     handler.writelines(('\n'.join(lst_data).encode('utf-8').decode('utf-8')))


import codecs

file = codecs.open("tiki.txt", "w+", "utf-8")
for item in lst_data:
    file.write(item.strip())
    file.write('\r\n')
file.close()