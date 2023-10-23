from bs4 import BeautifulSoup
html_doc = """<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""
soup = BeautifulSoup(html_doc,'html.parser')
# print(soup)
# print(soup.title)
# print(soup.a)
# print(soup.title.string)
# print(soup.a['id'])
# print(soup.find_all('a'))
# print(soup.find_all('p','story'))
# print(soup.find_all(id='link2'))
# print(soup.find_all(class_='sister'))
# print(soup.css.select('title'))

import requests,os
from urllib import request

url = 'https://www.google.com/search?q=%EA%B0%80%EC%9D%84&tbm=isch&ved=2ahUKEwi7_KjIvoGCAxXOyTQHHRsLAGoQ2-cCegQIABAA&oq=%EA%B0%80%EC%9D%84&gs_lcp=CgNpbWcQAzIECCMQJzILCAAQgAQQsQMQgwEyCAgAELEDEIMBMggIABCxAxCDATILCAAQgAQQsQMQgwEyCAgAELEDEIMBMggIABCxAxCDATILCAAQgAQQsQMQgwEyCAgAELEDEIMBMggIABCxAxCDAToFCAAQgAQ6CAgAEIAEELEDOgcIIxDqAhAnUOsFWO8dYKQfaAJwAHgBgAHcAYgB0g2SAQUwLjkuMZgBAKABAaoBC2d3cy13aXotaW1nsAEKwAEB&sclient=img&ei=U80wZfuGKs6T0-kPm5aA0AY&bih=865&biw=786'
result = requests.get(url)

path = os.path.dirname(__file__) + '/down/' 
if not os.path.exists(path):
    os.makedirs(path)

# print(result.text)
soup = BeautifulSoup(result.text,'html.parser')
imgdata = soup.find_all('img')
for i,item in enumerate(imgdata):
    try:
        request.urlretrieve(item['src'],path+'img_'+str(i)+'.jpg')
        print(item['src'])
    except:
        pass