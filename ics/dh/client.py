'''
Author : Shubham Sangamnerkar
Roll no : 4351

Kids : C++
Adults : Python
Legends : Sanskrit XD 
'''
import random
import socket
from init import p,q

b = random.getrandbits(32)

s = socket.socket()
s.connect(('127.0.0.1', 8000))

B = pow(q, b, p)
s.send(str(B).encode())

A = int(s.recv(20))
secret = pow(A, b, p)

print(secret)
s.close()
