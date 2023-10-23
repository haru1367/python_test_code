data = [
    ['홍길동','A',22],
    ['박정민','C',31],
    ['김철수','B',16],
    ['홍경인','C',23]
]

# data.sort(reverse=False,key=lambda x :(x[1],-x[2]))
# print(data)
sdata = sorted(data,key=lambda x :(x[1],-x[2]),reverse=False)
# print(data)
# print(sdata)

data1 =[
    {'name':'홍길동','score':'A','age':22},
    {'name':'박정민','score':'C','age':31},
    {'name':'김철수','score':'B','age':16},
    {'name':'홍경인','score':'C','age':23}
]
print(data1)
data1.sort(reverse=False,key=lambda x :x['score'])
print(data1)
sdata1 = sorted(data1,key=lambda x :(x['score'],-x['age']),reverse=False)
print(sdata1)
print(data1)