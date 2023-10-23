import namecardclass as ncc
import sys,pickle,os

'''
1.CardBook생성 2.Card추가 3. Card삭제 4.Card보기 5.전체목록 6.종료
프로그램 시작되면 데이터 파일을 로드
프로그램 종료되면 데이터 파일을 저장
'''
path = os.path.dirname(__file__)+'/cardbook.pickle'
cardbook = None
try:
    with open(path,'rb') as f:
        cardbook = pickle.load(f)
except (FileNotFoundError,EOFError) as e:
    print(e)

while True:
    menu = input('''                
1.CardBook생성 2.Card추가 3. Card삭제 4.Card보기 5.전체목록 6.종료
------------------------------------------------------------------
>>> ''')
    if menu == '1':
        if cardbook == None :
            cardbook = ncc.CardBook(input('cardbook 제목 >>>'))
        else:
            print('생성된 cardbook이 있습니다.')
    elif menu == '2':
        if cardbook == None:
            print('cardbook 먼저 생성하세요')
        else:
            name = input('name >>>')
            address = input('address >>>')
            tel = input('tel >>>')
            email = input('email >>>')
            card = ncc.Card(name,address,tel,email)
            cardbook.add_card(card)
            print(card)
            print(cardbook)
    elif menu == '3':
        if cardbook == None:
            print('cardbook 먼저 생성하세요')
        else:
            print(list(cardbook.cards.keys()))
            page = int(input('삭제할 page >>>'))
            if page in list(cardbook.cards.keys()):
                card = cardbook.remove_card(page)
                print(card,'-> 삭제')
            else:
                print('page를 잘못입력하셨습니다.')
    elif menu == '4':
        if cardbook == None:
            print('cardbook 먼저 생성하세요')
        else:
            print(list(cardbook.cards.keys()))
            page = int(input('검색할 page >>>'))
            if page in list(cardbook.cards.keys()):
                print(cardbook.cards[page])
            else:
                print('page를 잘못입력하셨습니다.')    
    elif menu == '5':
        if cardbook == None:
            print('cardbook 먼저 생성하세요')
        else:
            key = input('정렬키(입력값:name,address,tel,email)>>>')
            sort = bool(input('오름차순(enter),내림차순(1) >>>'))
            if key in ('name','address','tel','email'):
                cardbook.list_card(key,sort)
            else:
                print('키가 잘못됐습니다. 기본키로 정렬합니다.')
                cardbook.list_card(reverse=sort)
    elif menu == '6':
        print('프로그램 종료')
        with open(path,'wb') as f:
            pickle.dump(cardbook,f)
        sys.exit()
    else:
        print('메뉴선택을 잘못하셨습니다.')