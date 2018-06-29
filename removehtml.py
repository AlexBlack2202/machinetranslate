#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import unicodedata

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

# def no_accent_vietnamese(s):
#     s = s.decode('utf-8')
#     s = re.sub(u'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
#     s = re.sub(u'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
#     s = re.sub(u'èéẹẻẽêềếệểễể', 'e', s)
#     s = re.sub(u'ÈÉẸẺẼÊỀẾỆỂỄ', 'E', s)
#     s = re.sub(u'òóọỏõôồốộổỗơờớợởỡơờ', 'o', s)
#     s = re.sub(u'ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ', 'O', s)
#     s = re.sub(u'ìíịỉĩ', 'i', s)
#     s = re.sub(u'ÌÍỊỈĨ', 'I', s)
#     s = re.sub(u'ùúụủũưừứựửữư', 'u', s)
#     s = re.sub(u'ƯỪỨỰỬỮÙÚỤỦŨ', 'U', s)
#     s = re.sub(u'ỳýỵỷỹ', 'y', s)
#     s = re.sub(u'ỲÝỴỶỸ', 'Y', s)
#     s = re.sub(u'Đ', 'D', s)
#     s = re.sub(u'đ', 'd', s)
#     return s.encode('utf-8')

def no_accent_vietnamese(s):
    s = s.decode('utf-8')
    s = re.sub(u'Đ', 'D', s)
    s = re.sub(u'đ', 'd', s)
    return unicodedata.normalize('NFKD', unicode(s)).encode('ASCII', 'ignore')

def cleandata(data_name_file,data_clean_file,data_no_accent_file):

    with open(data_name_file,'r') as handler:
        content = handler.readlines()

    data_list =[]

    data_non_accent_list = []

    for item in content:
        data_clean = cleanhtml(item)
        data_clean  =data_clean.replace('"','')
        
        data_clean = data_clean.replace('.', ' .\n')# and new line for new sentence, and space before dot
        data_clean = data_clean.replace(',', ' ,\n')# and new line for new sentence, and space before dot
        data_clean = data_clean.replace('/', ' / ')
        data_clean = data_clean.replace('\n ', '\n')#remove space after \n
        data_clean = re.sub(' +',' ',data_clean) #remove multiple space

        data_clean_arr = data_clean.split('\n')

        data_list.extend([x.strip() for x in data_clean_arr if len(x)>5])

    for item in data_list:
        data_non_accent_list.append(no_accent_vietnamese(item))


    with open(data_clean_file,'w+') as handler:
        handler.writelines('\n'.join(data_list))

    with open(data_no_accent_file,'w+') as handler:
        handler.writelines('\n'.join(data_non_accent_list))


cleandata('data.csv','data_clean.txt','data_noaccent_clean.txt')
cleandata('data_test.csv','data_clean_test.txt','data_noaccent_clean_test.txt')