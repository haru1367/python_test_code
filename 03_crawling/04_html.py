# https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=0&ie=utf8&query=%EA%B0%80%EC%9D%84%EA%BD%83

from urllib import request,parse

query = input('검색어 >>>')
url = 'https://search.naver.com/search.naver?'

values ={
    # 'where':'nexearch',
    # 'sm':'top_hty',
    # 'fbm':0,
    # 'ie':'utf8',
    'query':query
}
querystring = parse.urlencode(values)
print(querystring)
result = request.urlopen(url+querystring).read().decode('utf8')
print(result)