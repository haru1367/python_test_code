import os
import namecarddef as ncf
# from namecarddef import *

card=[{'name':'홍길동', 'address':'서울', 'tel':'111-1111-1111', 'email':'hong@gmail.com'},
      {'name':'김나리', 'address':'광주', 'tel':'111-1111-1111', 'email':'kim@gmail.com'}]

path = os.path.dirname(__file__)+'/namecard1.json'
card = ncf.data_load(path)

while True:
    menu = input('''
-------------------------------------------------------
  1.입력  2.수정  3.삭제  4.목록  5.검색  6.저장  7.종료
-------------------------------------------------------
 >>> ''')
    if menu == '1':
        card = ncf.data_input(card)
    # 수정할 정보를 이름으로 찾아서 특정 항목만 수정(주소,전화번호,이메일중)
    elif menu == '2':
        card = ncf.data_update(card)
    elif menu == '3':
        card = ncf.data_delete(card)  
    elif menu == '4':
        ncf.data_list(card)        
    elif menu == '5':
        ncf.data_search(card)
    elif menu == '6':
        ncf.data_save(path,card)
    elif menu == '7':
        print('프로그램 종료!')   
        ncf.data_save(path,card)
        break
    else:
        print('메뉴선택이 잘못되었습니다.')


# 함수로 변경은 내일 작성
# 강의장에서 수정 작업 중