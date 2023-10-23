import json

card=[{'name':'홍길동', 'address':'서울', 'tel':'111-1111-1111', 'email':'hong@gmail.com'},
      {'name':'김나리', 'address':'광주', 'tel':'111-1111-1111', 'email':'kim@gmail.com'}]

with open('02_program/namecard1.json','r') as f:
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
                if item['name'] == name:
                    check = 1
            if check == 0:
                break
            print('중복된 이름이 있습니다.')
        address = input('주소 >>> ')
        tel = input('전화번호 >>> ')
        email = input('이메일 >>> ')
        card.append({'name':name, 'address':address, 'tel':tel, 'email':email})
        print(card) 
    # 수정할 정보를 이름으로 찾아서 특정 항목만 수정(주소,전화번호,이메일중)
    elif menu == '2':
        name = input('수정할 이름 >>> ')
        idx = -1
        for index, item in enumerate(card):
            if name == item['name']:
                idx = index
                while True:
                    update = input('수정할 항목(address,tel,email,exit(종료))>>>')
                    if update in ('address','tel','email'):
                        card[index][update] = input('수정내용 입력 >>> ')
                    elif update == 'exit':
                        break
                break
        if idx == -1:
            print('등록된 데이터가 없습니다.')        
    elif menu == '3':
        name = input('삭제할 이름 >>> ')
        delok = 0
        for index,item in enumerate(card):
            if item['name'] == name:
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
    이    름 : {item['name']}
    주    소 : {item['address']}
    전화번호 : {item['tel']}
    이 메 일 : {item['email']}''')
        
    elif menu == '5':
        name = input('검색할 이름 >>> ')
        idx = -1
        for index, item in enumerate(card):
            if item['name'] == name:
                idx = index
                print(f'''
    {index+1}page
--------------------------------
    이    름 : {item['name']}
    주    소 : {item['address']}
    전화번호 : {item['tel']}
    이 메 일 : {item['email']}''')
                break
        if idx == -1:
            print('등록되지 않은 데이터입니다.')
    elif menu == '6':
        with open('02_program/namecard1.json','w') as f:
            json.dump(card,f,indent=2)
    elif menu == '7':
        print('프로그램 종료!')   
        with open('02_program/namecard1.json','w') as f:
            json.dump(card,f,indent=2)
        break
    else:
        print('메뉴선택이 잘못되었습니다.')


# 함수로 변경은 내일 작성
# 강의장에서 수정 작업 중