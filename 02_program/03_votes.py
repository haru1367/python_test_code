'''
반장선거에서 투표한 후보자 번호들을 공백을 구분자로 하여 입력하면
각 값들을 분리하고 분리한 값들은 숫자로 변경하여 
각 숫자별로 같은 값의 갯수를 체크하여 출력하는 프로그램
함수는 아래와 같이 작성
- 공백을 구분자로 하여 입력받은 후보자 번호를 분리하는 함수
- 숫자로 변경된 값들의 항목별로 갯수를 카운트 하는 함수
- 결과값을 출력하는 함수
입력예 -> 1 2 3 5 3 4 2 1 2
출력예 -> 기호 : 1  득표수 : 2
          기호 : 2  득표수 : 3
          기호 5 번까지 
'''
def str2int(votes):
    # result = votes.split()
    # for index,item in enumerate(result):
    #     result[index] = int(item)
    return list(map(int,votes.split()))

def countvotes(votes):
    n = max(votes)
    print('가장큰값:',n)
    # result=[]
    # for i in range(n):
    #     result.append(0)
    result = [0 for i in range(n)]
    print(result)
    for item in votes:
        result[item-1] += 1
    return result

def printresult(votes):
    for index,item in enumerate(votes):
        print(f'기호 : { index+1 :2}  득표수 : {item}') 

votes = input('투표데이터 >>> ')
print(votes)
print(type(votes))
result = str2int(votes)
print(result)
print(type(result))
result = countvotes(result)
print(result)
printresult(result)
