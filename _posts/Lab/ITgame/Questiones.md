[toc]

# 1. 题目：四个数字：1、2、3、4，能组成多少互不相同且无重复数字的三位数？各是多少？
程序分析：可填在百位、十位、个位的数字都是1、2、3、4。组成所有的排列后再去掉不满足条件的排列。

```py
1.
a=0
for i in range(1,5):
    for j in range(1,5):
        for k in range(1,5):
            if( i != k ) and (i != j) and (j != k):
                a+=1
                print i,j,k
print "amount:", i
# 结果
1 2 3
1 2 4
1 3 2
1 3 4
...
amount: 24


1.1
list_num=['1','2','3','4']
list=[]
turns=0
for i in list_num:
    for j in list_num:
         for k in list_num:
            if( i != k ) and (i != j) and (j != k):
                turns+=1
                list.append(int(i+j+k))
                # cannot be list += int(i+j+k)
                # TypeError: 'int' object is not iterable.
print list
print "amount is:", turns
# 结果
[123, 124, 132, 134, 142, 143, 213, 214, 231, 234, 241, 243, 312, 314, 321, 324, 341, 342, 412, 413, 421, 423, 431, 432]
amount: 24


1.2 better
list_num=['1','2','3','4']
list_result=[]
for i in list_num:
    for j in list_num:
        for k in list_num:
            if len(set(i+j+k))==3:        #always true and set the num
                list_result+=[int(i+j+k)]
print "amount is:", len(list_result)
print list_result

# ---------------------------------------------------------

2. best
list_num=[1,2,3,4]
list = [i*100 + j*10 + k for i in list_num for j in list_num for k in list_num if (j != i and k != j and k != i)]
print 'amount is: %s' % len(list)
print 'they are : %s' % list
# 结果
amount is: 24
[123, 124, 132, 134, 142, 143, 213, 214, 231, 234, 241, 243, 312, 314, 321, 324, 341, 342, 412, 413, 421, 423, 431, 432]

2.1
#直接用列表推导式
a=[(x,y,z) for x in range(1,5) for y in range(1,5) for z in range(1,5) if(x!=y)and(x!=z)and(y!=z)]
print a
# 结果
[(1, 2, 3), (1, 2, 4), (1, 3, 2)...]
# ---------------------------------------------------------

3. python自带这个函数的
from itertools import permutations
for i in permutations([1, 2, 3, 4], 3):
    print(i)
# 结果
(1, 2, 3)
(1, 2, 4)
(1, 3, 2)
for i in permutations('1234',3):
    print i
# 结果
('1', '2', '3')
('1', '2', '4')
('1', '3', '2')

3.1
from itertools import permutations
a=[]
for i in permutations([1, 2, 3, 4], 3):
    a.append(i)
print a
# 结果
[(1, 2, 3), (1, 2, 4), (1, 3, 2), (1, 3, 4), (1, 4, 2), (1, 4, 3), (2, 1, 3), (2, 1, 4), (2, 3, 1), (2, 3, 4), (2, 4, 1), (2, 4, 3), (3, 1, 2), (3, 1, 4), (3, 2, 1), (3, 2, 4), (3, 4, 1), (3, 4, 2), (4, 1, 2), (4, 1, 3), (4, 2, 1), (4, 2, 3), (4, 3, 1), (4, 3, 2)]

3.2
from itertools import permutations
a=[]
b=[]
for i in permutations(['1','2','3','4'], 3):   # not int, be string
    a.append(i)
for x,y,z in a:
    b.append(int(x+y+z))
print b
print len(a)
# 结果
[123, 124, 132, 134, 142, 143, 213, 214, 231, 234, 241, 243, 312, 314, 321, 324, 341, 342, 412, 413, 421, 423, 431, 432]
24


3.3
from itertools import permutations
a=[]
for i in permutations([1, 2, 3, 4], 3):
    k = ''
    for j in range(0, len(i)):
        k = k + str(i[j])
    a.append(k)
    print (int(k))
    # 123
    # 124
    # 132
    # 134
    # ...
print a
# ['123', '124', '132', '134', '142', '143', '213', '214', '231', '234', '241', '243', '312', '314', '321', '324', '341', '342', '412', '413', '421', '423', '431', '432']


4.
from itertools import permutations
t = 0
for i in permutations('1234',3):
# i =
# ('1', '2', '3')
# ('1', '2', '4')
    print(''.join(i))
    t += 1
print("amount:%s"%t)
```

---

# 2. 题目：企业发放的奖金根据利润提成
- 企业发放的奖金根据利润提成。
  - 利润(I)低于或等于10万元时，奖金可提10%；
  - 利润高于10万元，低于20万元时，低于10万元的部分按10%提成，高于10万元的部分，可提成7.5%；
  - 20万到40万之间时，高于20万元的部分，可提成5%；
  - 40万到60万之间时高于40万元的部分，可提成3%；
  - 60万到100万之间时，高于60万元的部分，可提成1.5%，
  - 高于100万元时，超过100万元的部分按1%提成，
  - 从键盘输入当月利润I，求应发放奖金总数？
- 程序分析：请利用数轴来分界，定位。注意定义时需把奖金定义成长整型。

```py
i=int(raw_input('Input your benefits:'))
arr = [1000000,600000,400000,200000,100000,0]
BonusRate = [0.01,0.015,0.03,0.05,0.075,0.1]
Bonus=0
for index in range(0,6):       # or range(len(arr))
    if i > arr[index]:
      Bonus+=(i-arr[index])*BonusRate[index]
      print 'gained bonus: ', (i-arr[index])*BonusRate[index]
      i=arr[index]
print 'total bonus:', Bonus
```

# 3. 题目：一个整数，它加上100后是一个完全平方数，再加上168又是一个完全平方数，请问该数是多少？

- 程序分析：
  - 假设该数为 x。
    1. x + 100 = n^2, x + 100 + 168 = m^2
    2. 计算等式：m^2 - n^2 = (m + n)(m - n) = 168
    3. 设置： m + n = i，m - n = j，i * j =168，i 和 j 至少一个是偶数
    4. 可得： m = (i + j) / 2， n = (i - j) / 2，i 和 j 要么都是偶数，要么都是奇数。
    5. 从 3 和 4 推导可知道，i 与 j 均是大于等于 2 的偶数。
    6. 由于 i * j = 168， j>=2，则 1 < i < 168 / 2 + 1。

# 4. 题目：输入某年某月某日，判断这一天是这一年的第几天？
- 程序分析：以3月5日为例，应该先把前两个月的加起来，然后再加上5天即本年的第几天，特殊情况，闰年且输入月份大于2时需考虑多加一天：

```py

1.
year = int(raw_input('year:'))
month = int(raw_input('month:'))
day = int(raw_input('day:'))
months = (0,31,59,90,120,151,181,212,243,273,304,334)

if 0 < month <= 12:
  Mday=0
  for i in range(len(months)):
    if month-1 == i:
      Mday+=int(months[i])
  if (year%4 == 0) and (month>2):
    print 'it is the %dth day.' % int(Mday+day)+1
  else:
    print 'it is the %dth day.' % int(Mday+day)
else: print 'data error'

1.1
year = int(raw_input('year:'))
month = int(raw_input('month:'))
day = int(raw_input('day:'))
months1=[0,31,60,91,121,152,182,213,244,274,305,335,366] #28
months2=[0,31,59,90,120,151,181,212,243,273,304,334,365] #30

if 0 < month <= 12:
    Mday=0
    if (year%4 == 0) and (month>2):
        Dth=day+months1[month-1]
        print 'it is the %dth day.' % Dth
    else:
        Dth=day+months2[month-1]
        print 'it is the %dth day.' % Dth
else: print 'data error'

2.
import time

D=raw_input("input date like (XXXX-XX-XX): ")
d=time.strptime( D,'%Y-%m-%d').tm_yday
print "the {} day of this year!" .format(d)

2.1
import time
date = raw_input("input date like (XXXX-XX-XX): ")
print(time.strptime(date, '%Y-%m-%d')[7])


3.
from functools import reduce

year = int(input('the year:'))
month = int(input('the month:'))
day = int(input('the day:'))

mday = [0,31,28 if year%4 else 29 if year%100 else 28 if year%4 else 29,31,30,31,30,31,31,30,31,30,31]

print('{}.{}.{} is the {}th day.'.format(year, month, day, reduce(lambda x,y:x+y, mday[:month])+day))
```

---

# 5. 题目：输入三个整数x,y,z，请把这三个数由小到大输出。
- 程序分析：我们想办法把最小的数放到x上，先将x与y进行比较，如果x>y则将x与y的值进行交换，然后再用x与z进行比较，如果x>z则将x与z的值进行交换，这样能使x最小。
