'''
Author : Shubham Sangamnerkar
Roll no : 4351

Kids : C++
Adults : Python
Legends : Sanskrit XD 


References : https://en.wikipedia.org/wiki/Tonelli%E2%80%93Shanks_algorithm
             https://en.wikipedia.org/wiki/Euler%27s_criterion
             

'''
import random
class gV:
    #initialization by default values, input may overwrite these default values
    a = 0
    b = 7
    n = 137
    G = [1, 75]
    na = None
    nb = None
    pa = None
    pb = None

def eulers_criterion(n, p):
    return pow(n, (p-1) // 2, p) != p-1

def shank_tonelli(n, p):
    if n == 0 or p == 2:
        return [n]
    #Check if eulers criterion is satisfied
    if eulers_criterion(n, p) == False:
        return []
    #reduce p-1 to form (s * 2^e)
    e = 0
    s = p - 1
    while s % 2 == 0:
        e += 1
        s //= 2
    #finding smallest q such that eulers_criterion(q, p) == p-1
    q = 1
    while eulers_criterion(q, p):
        q += 1
    #initialize x, b, g
    x = pow(n, (s+1) // 2, p)
    t = pow(n, s, p)
    c = pow(q, s, p)
    m = e
    
    while t != 1:
        #find smallest i such that t**(2**i) % p = 1
        i, it = 0, 2
        for i in range(1, m):
            if pow(t, it, p) == 1:
                break
            it *= 2
        #make updations
        b = pow(c, 2 ** (m - i - 1), p)
        x = (x * b) % p
        t = (t * b * b) % p
        c = (b * b) % p
        m = i
    return [x, p-x]

def generate_point(a, b, n):
    if 4 * (a ** 3) + 27 * (b ** 2)==0:
        print("Singularity violated")
    else:
        x = 1
        while True:
            if x > 10 ** 6:
                print("G does not exist for the given a, b, n in the finite space")
                exit(1)
            rhs = ( x**3 + a * x + b) % n
            lhs = shank_tonelli(rhs, n)
            if len(lhs) > 0:
                return [x, lhs[0]]
            else:
                x += 1

def input_a_b_n():
    gV.a = int(input("Enter a : "))
    gV.b = int(input("Enter b : "))
    gV.n = int(input("Enter n : "))

def initialize_na_nb_G():
    gV.na = 13 #private key of alice
    gV.nb = 15 #private key for bob
    gV.G = generate_point(gV.a, gV.b, gV.n)

def find_public_keys():
    gV.pa = [gV.na * gV.G[0], gV.na * gV.G[1]]
    gV.pb = [gV.nb * gV.G[0], gV.nb * gV.G[1]]
    print("Public key of alice : {}".format(gV.na))
    print("Public key of bob : {}".format(gV.nb))
    print("Generator point : {}".format(gV.G))

def encrypt(m):
    k = random.randint(0, 10)
    c1 = (k * (gV.G[0] + gV.G[1])) % gV.n
    c2 = (m + (k * gV.pb[0] + k * gV.pb[1])) % gV.n
    return [c1, c2]
    
def decrypt(C):
    return ((C[1] - C[0] * gV.nb) % gV.n + gV.n) % gV.n

input_a_b_n()
initialize_na_nb_G()
find_public_keys()
m = int(input("Enter plaintext integer : "))
if m > gV.n:
    print("Point out of finite space")
cipher_text = encrypt(m)
print("Cipher text : {}".format(cipher_text))
print("Decrypted text : {}".format(decrypt(cipher_text)))

