from urllib import request
import os
url = 'https://www.urbanbrush.net/web/wp-content/uploads/edd/2022/11/urbanbrush-20221108214712319041.jpg'
path = os.path.dirname(__file__) + '/down/' 
if not os.path.exists(path):
    os.makedirs(path)
filename = path + url.split('/')[-1]
request.urlretrieve(url,filename)
