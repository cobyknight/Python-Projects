import array as arr
import random
import numpy as np

print('input total number of stats')
x = int(input())

intRolls = np.array([])
finArray = np.array([])

w = 0
while w <= 5:
    
    i = 0
    while i <=3 :
        randomNum = random.randint(1,6)
        intRolls = np.insert(intRolls, i, randomNum)
        intRolls.sort()
        i += 1
    
    intRolls = intRolls[1:]
    print(intRolls)
    
    num = np.sum(intRolls)
    print(num)
    
    intRolls = []
    
    
    finArray = np.insert(finArray, w, num)
    
    finalSum = finArray.sum()

    #print(finArray)
    
    w += 1

extraPoints = finalSum - finalSum * (1 - (x - finalSum)/finalSum)

finArray.sort()
finArray = np.insert(finArray, 6, extraPoints)

print(finArray)