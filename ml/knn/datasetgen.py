'''
Author : Shubham Sangamnerkar
Roll no : 4351

Kids : C++
Adults : Python
Legends : Sanskrit XD 

to generate dataset run

python3 datasetgen.py > dataset.csv 
in terminal

'''
import random

print("X,Y,Result")
for i in range(1000):
    x = random.randint(0, 100)
    y = random.randint(0, 100)
    z = random.randint(0,1)
    print("{},{},{}".format(x,y,z))


        
        