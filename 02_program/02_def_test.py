# def add(a,b):
#     return a+b

# a = int(input('숫자 >>> '))
# b = int(input('숫자 >>> '))

# print(add(a,b))

# 입력값 x, 출력값 x
# def add():
#     a = int(input('숫자>>>'))
#     b = int(input('숫자 >>> '))
#     print(a+b)

# add()
# 입력값 o, 리턴값 x
# def add(a,b):
#     print(a+b)

# a = int(input('숫자>>>'))
# b = int(input('숫자 >>> '))
# add(b=a,a=8)

# 입력값이 몇개가 될지 모를때
# def add(*a,choice='sub'):
#     print(a)
#     print(type(a))
#     print(choice)
#     result = 0
#     for i in a:
#         result += i
#     return result

# print(add(1,2,3,4,5,choice='add'))
# print(add(2,4,8,9,7))

# 키워드 매개변수

# def print_kwargs(**a):
#     print(a)
#     print(type(a))

# print_kwargs(name='hong',age=22,city='busan')

# 함수의 리턴값은 하나
# def add_and_mul(a,b):
#     return a+b,a*b

# print(type(add_and_mul(1,2)))
# print(add_and_mul(1,2))

# a = 1
# def vartest(a):
#     a = a + 1
#     print('함수안',a)
#     return a
# a = vartest(a)
# print(a)

import os
print(__file__)
print(os.path.dirname(__file__))