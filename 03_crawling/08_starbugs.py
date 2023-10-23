from bs4 import BeautifulSoup
import requests

url = 'https://www.starbucks.co.kr/store/store_map.do?disp=quick'
res = requests.get(url).text
soup = BeautifulSoup(res,'html.parser')
# print(soup)
print(soup.select('.quickSearchResultBox strong'))

from selenium import webdriver
import time
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get(url)
time.sleep(5)
page = driver.page_source
soup = BeautifulSoup(page,'html.parser')
# print(soup)
print(soup.select('.quickSearchResultBox strong')[0].string)
print(soup.select_one('li.quickResultLstCon p').text)

# .loca_search a 클릭
loca = driver.find_element(By.CSS_SELECTOR,'.loca_search a')
print(type(loca))
print(loca)
loca.click()
time.sleep(5)
# .sido_arae_box
sido = driver.find_elements(By.CSS_SELECTOR,'.sido_arae_box li')
print(sido)
for index, item in enumerate(sido):
    print(index, ':',item.text,end='  ')
sido[16].click()
time.sleep(5)
page = driver.page_source
soup = BeautifulSoup(page,'html.parser')
# print(soup)
print(soup.select('.quickSearchResultBox strong')[0].string)
print(soup.select('li.quickResultLstCon p')[0].text)


