import sys,os,json

class Namecard:
    def __init__(self) -> None:
        self.card=[]
        self.path = os.path.dirname(__file__)+'/namecard1.json'
        self.data_load()
        while True:
            self.exe(self.display())

    def display(self):    
        menu = input('''
-------------------------------------------------------
  1.입력  2.수정  3.삭제  4.목록  5.검색  6.저장  7.종료
-------------------------------------------------------
 >>> ''')
        return menu
    
    def exe(self,menu):
        if menu == '1':
            self.data_input()
        # 수정할 정보를 이름으로 찾아서 특정 항목만 수정(주소,전화번호,이메일중)
        elif menu == '2':
            self.data_update()
        elif menu == '3':
            self.data_delete()  
        elif menu == '4':
            self.data_list()        
        elif menu == '5':
            self.data_search()
        elif menu == '6':
            self.data_save()
        elif menu == '7':
            print('프로그램 종료!')   
            self.data_save()
            sys.exit()
        else:
            print('메뉴선택이 잘못되었습니다.')

    def data_load(self):
        with open(self.path,'r') as f:
            self.card = json.load(f)

    def data_save(self):
        with open(self.path,'w') as f:
            json.dump(self.card,f,indent=2)

    def data_input(self):
        while True:
            name = input('이름 >>> ')
            check = 0
            for item in self.card:
                if item['name'] == name:
                    check = 1
            if check == 0:
                break
            print('중복된 이름이 있습니다.')
        address = input('주소 >>> ')
        tel = input('전화번호 >>> ')
        email = input('이메일 >>> ')
        self.card.append({'name':name, 'address':address, 'tel':tel, 'email':email})
        print(self.card) 
  
    def data_update(self):
        name = input('수정할 이름 >>> ')
        idx = -1
        for index, item in enumerate(self.card):
            if name == item['name']:
                idx = index
                while True:
                    update = input('수정할 항목(address,tel,email,exit(종료))>>>')
                    if update in ('address','tel','email'):
                        self.card[index][update] = input('수정내용 입력 >>> ')
                    elif update == 'exit':
                        break
                break
        if idx == -1:
            print('등록된 데이터가 없습니다.')

    def data_delete(self):
        name = input('삭제할 이름 >>> ')
        delok = 0
        for index,item in enumerate(self.card):
            if item['name'] == name:
                print(item,'삭제~')
                del self.card[index]
                delok = 1
                break
        if delok == 0:
            print('등록되지 않은 데이터입니다.')  

    def data_list(self):
        for index , item in enumerate(self.card):    
            print(f'''
        {index+1}번째
    --------------------------------
        이    름 : {item['name']}
        주    소 : {item['address']}
        전화번호 : {item['tel']}
        이 메 일 : {item['email']}''')

    def data_search(self):
        name = input('검색할 이름 >>> ')
        idx = -1
        for index, item in enumerate(self.card):
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

if __name__ == '__main__':
    Namecard()