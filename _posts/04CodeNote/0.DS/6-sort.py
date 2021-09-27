

alist = [54,26, 93,17, 77,31, 44,55, 20]
# alist = [19, 1, 9, 7, 3, 10, 13, 15, 8, 12] 
# alist = [18, 54,296,13,17]
n = len(alist)

def bubbleSort(alist):
    print(alist)
    compare_num = 0
    change_num = 0
    for passnum in range(len(alist)-1,0,-1):
        # print(alist[n])
        for i in range(passnum):
            compare_num += 1
            if alist[i]>alist[i+1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp
                change_num+=1
        # print(alist)
    print("compare_num", compare_num)
    print("change_num", change_num)
        
print(" =============== bubbleSort ")
compare_pre=n*n/2 - n/2
print("n:", n, " time: n^2/2 - n/2", compare_pre)
bubbleSort(alist)
# print(alist)


def selectionSort(alist):
    compare_num=0
    change_num=0
    for fillslot in range(len(alist)-1,0,-1):
        positionOfMax=0
        for location in range(1,fillslot+1):
                # print(alist)
                # print(alist[location])
                compare_num += 1
                if alist[location]>alist[positionOfMax]:
                    positionOfMax = location
        temp = alist[fillslot]
        alist[fillslot] = alist[positionOfMax]
        alist[positionOfMax] = temp
        change_num+=1
    print("compare_num", compare_num)
    print("change_num", change_num)

print(" =============== selectionSort ")
compare_pre=n*n/2 - n/2
print("n:", n, " time: n^2/2 - n/2", compare_pre)
selectionSort(alist)
# print(alist)

for i in range(1,3):
    print(i)