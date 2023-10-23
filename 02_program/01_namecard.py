# 명함관리 프로그램
# 기능(입력,수정,삭제,목록,검색,종료,저장)
# 데이터구조(이름,주소,전화번호,이메일)- 리스트
# 데이타는 이름을 키값으로 
# 중복된 이름은 입력 안됨.

import json,os

card = [['홍길동','세종','010-111-2222','hong@gmail.com'],
        ['김태우','세종','010-2222-3333','kim@gmail.com']]

with open('02_program/namecard.json','r') as f:
    card = json.load(f)

while True:
    menu = input('''
-------------------------------------------------------
  1.입력  2.수정  3.삭제  4.목록  5.검색  6.저장  7.종료
-------------------------------------------------------
 >>> ''')
    if menu == '1':
        while True:
            name = input('이름 >>> ')
            check = 0
            for item in card:
                if item[0] == name:
                    check = 1
            if check == 0:
                break
            print('중복된 이름이 있습니다.')
        address = input('주소 >>> ')
        tel = input('전화번호 >>> ')
        email = input('이메일 >>> ')
        card.append([name,address,tel,email])
        print(card) 
    # 수정할 정보를 이름으로 찾아서 특정 항목만 수정(주소,전화번호,이메일중)
    elif menu == '2':
        name = input('수정할 이름 >>> ')
        idx = -1
        for index, item in enumerate(card):
            if name == item[0]:
                idx = index
                update = int(input('수정할 항목(1.주소 2.전화번호 3.이메일)>>>'))
                card[index][update] = input('수정내용 입력 >>> ')
                break
        if idx == -1:
            print('등록된 데이터가 없습니다.')        
    elif menu == '3':
        name = input('삭제할 이름 >>> ')
        delok = 0
        for index,item in enumerate(card):
            if item[0] == name:
                print(item,'삭제~')
                del card[index]
                delok = 1
                break
        if delok == 0:
            print('등록되지 않은 데이터입니다.')    
    elif menu == '4':
        for index , item in enumerate(card):    
            print(f'''
    {index+1}번째
--------------------------------
    이    름 : {item[0]}
    주    소 : {item[1]}
    전화번호 : {item[2]}
    이 메 일 : {item[3]}''')
        
    elif menu == '5':
        name = input('검색할 이름 >>> ')
        idx = -1
        for index, item in enumerate(card):
            if item[0] == name:
                idx = index
                print(f'''
    {index+1}page
--------------------------------
    이    름 : {item[0]}
    주    소 : {item[1]}
    전화번호 : {item[2]}
    이 메 일 : {item[3]}''')
                break
        if idx == -1:
            print('등록되지 않은 데이터입니다.')
    elif menu == '6':
        with open('02_program/namecard.json','w') as f:
            json.dump(card,f,indent=2)
    elif menu == '7':
        print('프로그램 종료!')   
        with open('02_program/namecard.json','w') as f:
            json.dump(card,f,indent=2)
        break
    else:
        print('메뉴선택이 잘못되었습니다.')
