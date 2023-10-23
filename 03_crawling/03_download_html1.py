import requests

url = 'https://www.naver.com'
result = requests.get(url)
print(result)
print(type(result))
print(result.text)
print(result.encoding)

r = requests.get('https://api.github.com/events')
print(r.text)
print(r.json())