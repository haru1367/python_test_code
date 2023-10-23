import requests
from bs4 import BeautifulSoup

url ='https://finance.naver.com/marketindex/exchangeList.naver'
result = requests.get(url).text
# print(result)
soup = BeautifulSoup(result,'html.parser')
name=[]
price=[]
data = soup.select('td.tit > a')
# print(data)
for item in data:
    # print(item.text.strip())
    name.append(item.text.strip())
print(name)
data = soup.select('td.sale')
# print(data)
for item in data:
    # print(item.text.strip().replace(',',''))
    price.append(float(item.text.strip().replace(',','')))
print(price)
print(len(name),len(price))

result = list(zip(name,price))
print(result)