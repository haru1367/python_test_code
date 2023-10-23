from urllib import request
import json

url = 'https://www.naver.com'
result = request.urlopen(url).read().decode('utf8')
print(result)
print(type(result))

url = 'https://api.github.com/events'
result = request.urlopen(url).read()
print(json.loads(result))