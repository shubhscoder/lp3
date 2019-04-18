'''
Author : Shubham Sangamnerkar
Roll no : 4351

Kids : C++
Adults : Python
Legends : Sanskrit XD 
'''

key = [0,1,1,1,1,1,1,1,0,1]
p10 = [3,5,2,7,4,10,1,9,8,6]
p8 = [6,3,7,4,8,5,10,9]
p4 = [2,4,3,1]
ip = [2,6,3,1,4,8,5,7]
EP = [4,1,2,3,2,3,4,1]
s0 = [[1,0,3,2],[3,2,1,0],[0,2,1,3],[3,1,3,2]]
s1 = [[0,1,2,3],[2,0,1,3],[3,0,1,0],[2,1,0,3]]
ip_1 = [4,1,3,5,7,2,8,6]
plain = input("Enter 8 bit plain text : ")
plain = [int(i) for i in plain]

def permutation(p, key):
    return [key[i-1] for i in p]

def shift(key, shifts):
    return key[shifts:] + key[:shifts]

def generate_key(key, shifts):
    leng = len(key) // 2
    return shift(key[:leng], shifts) + shift(key[leng:], shifts)

def xor(key1, key2):
    l = []
    for i in range(len(key1)):
        l.append(key1[i] ^ key2[i])
    return l

def sbox(key, sbox):
    row = int(str(key[0])+str(key[3]), 2)
    col = int(str(key[1])+str(key[2]), 2)
    return bin(sbox[row][col])[2:].zfill(2)

def encrypt_decrypt(k, plain):
    ip_plain = permutation(ip, plain)
    for i in range(len(k)):
        leng = len(ip_plain) // 2
        ep = permutation(EP, ip_plain[leng:])
        xor_ans = xor(ep, k[i])
        len_xor = len(xor_ans) // 2
        str_ans = sbox(xor_ans[:len_xor], s0) + sbox(xor_ans[len_xor:], s1)
        str_ans = [int(c) for c in str_ans]
        ip_plain = ip_plain[leng:] + xor(ip_plain[:leng], permutation(p4, str_ans))
    ip_plain = ip_plain[leng:] + ip_plain[:leng]
    final_ans = permutation(ip_1, ip_plain)
    print(final_ans)
    return final_ans

k = []
k.append(permutation(p8, generate_key(permutation(p10, key), 1)))
k.append(permutation(p8, generate_key(permutation(p10, key), 3)))
print(k)

cipher_text = encrypt_decrypt(k, plain)
k.reverse()
encrypt_decrypt(k, cipher_text)