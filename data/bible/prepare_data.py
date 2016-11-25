'''
Takes the html files and strips away headers and tags leving just the body
'''
import os
from os import listdir
from os.path import isfile, join
from bs4 import BeautifulSoup
onlyfiles = [join('./t', f) for f in listdir('./t') if isfile(join('./t', f))]


from bs4 import BeautifulSoup
txt =''

for num,f_path in enumerate(onlyfiles):
    print(num)
    f = open(f_path,encoding='Windows-1255')
    try:
        soup = BeautifulSoup(f, 'html.parser')
        if soup.body:
            txt += (soup.body.text)
    except:
        print(f_path)
    f.close()
f = open('./input.txt','w',encoding='utf-8')
f.write(txt)
f.close()




