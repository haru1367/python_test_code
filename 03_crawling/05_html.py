import requests

query = input('검색어 >>>')
url = 'https://search.naver.com/search.naver?'

values ={
    # 'where':'nexearch',
    # 'sm':'top_hty',
    # 'fbm':0,
    # 'ie':'utf8',
    'query':query
}
result = requests.get(url,params=values)
print(result.text)