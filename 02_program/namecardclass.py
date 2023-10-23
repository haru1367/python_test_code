'''
생성자작성 : name,address,tel,email을 입력받아서 객체 인스턴스를 작성한다.
card에 내용을 적는 함수 : write_card()
card에 내용을 삭제하는 함수 : remove_all()
card객체 인스턴스를 출력하면 
   name:이름, adress:주소, tel:전화번호, email:이메일
'''
class Card: # 명함 1장
    def __init__(self,name='',address='',tel='',email='') -> None:
        self.name=name
        self.address=address
        self.tel=tel
        self.email=email

    def write_card(self,name='',address='',tel='',email=''):
        if not name=='': self.name=name
        if not address=='': self.address=address
        if not tel=='':self.tel=tel
        if not email=='':self.email=email

    def remove_all(self):
        self.name=''
        self.address=''
        self.tel=''
        self.email=''

    def __str__(self) -> str:
        return f'name:{self.name}, address:{self.address}, tel:{self.tel}, email:{self.email}'

'''
전체 300장의 명함을 관리할수 있는 명함집 클래스 생성, 
card 객체 인스턴스는 딕셔너리로 관리한다. 키는 page_number를 사용
생성자 : 명함의 제목(title)을 입력받아서 명함집을 만든다. 
이때 명함의 전체 갯수값을 가지는 변수 page_number도 초기화 한다.
card를 명함집에 추가하는 함수 : add_card() 추가시 page_number값 증가
card를 명함집에서 제거하는 함수 : remove_card() 삭제시 page_number값 감소
명함집에 명함에 몇개 있는지 알려주는 함수 : get_number_of_pages():
전체 명함을 출력하는 함수:list_card()
   
'''
class CardBook: # 명함을 관리하는 클래스
    def __init__(self,title) -> None:
        self.title = title
        self.page_number = 1
        self.cards = {}

    def add_card(self,card,page=0):
        if self.page_number < 300:
            if page == 0:
                self.cards[self.page_number] = card
            else:
                self.cards[page] = card
            self.page_number += 1    
        else:
            print('페이지가 모두 채워졌습니다.')

    def remove_card(self,page):
        if page in self.cards.keys():
            return self.cards.pop(page)
        else:
            print('해당 페이지가 없습니다.')

    def get_number_of_pages(self):
        return len(self.cards)

    def list_card(self,key=None,reverse=False):
        if key == None:
            for page in self.cards:
                print(page,':',self.cards[page])
        else:
            sorted_data = sorted(self.cards.items(),
                                 key=eval(f'lambda item:item[1].{key}'),
                                 reverse=reverse)
            for page,card in sorted_data:
                print(page,':',card)

# print(__name__)
# print('if문 밖에 namecardclass.py에서 실행')
if __name__ == '__main__':
    # print('if문 안에 namecardclass.py에서 실행')
    # card = Card('홍길동','세종','000-888-9999')
    # card1 = Card(name='박문수',email='park@gmail.com')
    # print(card)
    # print(card1)
    # card.write_card('홍길순')
    # print(card)
    # card.remove_all()
    # print(card)
    book = CardBook('거래처')
    card = Card('홍길동','세종','000-888-9999')
    card1 = Card(name='박문수',address='부산',email='park@gmail.com')
    card2 = Card(name='김정남',address='대전',email='park@gmail.com')
    book.add_card(card)
    book.add_card(card1,5)
    book.add_card(card2,10)
    print(book.get_number_of_pages())
    book.list_card()
    print('-'*30)
    book.list_card('address')
    print('-'*30)
    book.list_card('address',True)
