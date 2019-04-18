'''
Author : Shubham Sangamnerkar
Roll no : 4351

Kids : C++
Adults : Python
Legends : Sanskrit XD 
'''

import random
import socket
from init import p, q

s = socket.socket()
s.bind(('', 8000))
s.listen()

try:
    while True:
        a = random.getrandbits(32)
        conn, addr = s.accept()
        
        A = pow(q, a, p)
        conn.send(str(A).encode())
        
        B = int(conn.recv(20))
        secret = pow(B, a, p)
        
        print(secret)
        conn.close()

except KeyboardInterrupt:
    s.close()