import json

def data_load(path):
    with open(path,'r') as f:
        return json.load(f)

def data_save(path,card):
    with open(path,'w') as f:
        json.dump(card,f,indent=2)

def data_input(card):
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
    return card

def data_update(card):
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
    return card

def data_delete(card):
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
    return card

def data_list(card):
    for index , item in enumerate(card):    
        print(f'''
    {index+1}번째
--------------------------------
    이    름 : {item['name']}
    주    소 : {item['address']}
    전화번호 : {item['tel']}
    이 메 일 : {item['email']}''')

def data_search(card):
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