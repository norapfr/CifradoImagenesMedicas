import numpy as np
import numpy.typing as npt
import cv2 as cv
import json
from typing import Tuple

#chaotic map
def cat_map_4d_matrix(a:int, b:int, c:int, d:int) -> npt.NDArray:
    A11=(2*b+c)*(d+1)+3*d+1
    A12=2*(b+1)
    A13=2*b*c+c+3
    A14=4*b+3
    E=2*(a+1)*(d+b*c*(d+1))+a*(c+1)*(d+1)
    A22=2*(a+b+a*b)+1
    F=a*(c+3)+2*b*c*(a+1)+2
    A24=3*a+4*b*(a+1)+1
    A31=3*b*c*(d+1)+3*d
    A32=3*b+1
    A33=3*b*c+3
    A34=6*b+1
    A41=c*(b+1)*(d+1)+d
    A42=b+1
    A43=b*c+c+1
    A44=2*b+2
    A=np.array([
        [A11,A12,A13,A14],
        [E,A22,F,A24],
        [A31,A32,A33,A34],
        [A41,A42,A43,A44]
    ], dtype=float)
    return A

def cat_map_2d_matrix(a:int, b:int) -> npt.NDArray:
    A=np.array([[1,a],[b,a*b+1]], dtype=float)
    return A

#keys
def generate_A():
    a1,b1 = np.random.randint(1,10,2)
    a2,b2 = np.random.randint(1,10,2)
    info={
        "A4": cat_map_4d_matrix(2,1,2,2),
        "A2_primary": cat_map_2d_matrix(a1,b1),
        "A2_secondary": cat_map_2d_matrix(a2,b2)
    }
    return info

def generate_X0():
    def rand_nonzero(n): 
        x=np.random.random(n)
        while np.allclose(x,0): x=np.random.random(n)
        return x
    return {
        "X0_A4": rand_nonzero(4),
        "X0_primary": rand_nonzero(2),
        "X0_secondary": rand_nonzero(2)
    }

def save_keys(A,X0):
    keys = {"A":{k:v.tolist() for k,v in A.items()}, "X0":{k:v.tolist() for k,v in X0.items()}}
    with open("keys.json","w") as f: json.dump(keys,f,indent=4)

def load_keys():
    with open("keys.json","r") as f: data=json.load(f)
    A  = {k:np.array(v) for k,v in data["A"].items()}
    X0 = {k:np.array(v) for k,v in data["X0"].items()}
    return A,X0

#prng
def prng_4d(A:npt.NDArray, X0:npt.NDArray, L:int):
    X = X0.copy()
    Y_list=[]
    U = np.array([2,3,5,1],dtype=float)
    iterations = L // 16 + 1
    for _ in range(iterations):
        t_vec = np.dot(U,X)
        t = int(np.ceil(t_vec))+1
        for _ in range(t):
            X = np.dot(A,X) % 1.0
        Y_list.append(X.copy())
    Y=np.concatenate(Y_list)[:L]
    return Y

def get_omega(A, X0, l, n):
    L=l*n
    Y=prng_4d(A,X0,L)
    Z=(np.floor((2**32)*Y)).astype(np.uint32)
    Z_bytes = Z.view(np.uint8)
    omega = Z_bytes[:L].reshape((l,n))
    return omega

def get_psi(A,X0,r,l):
    L=l*r
    Y=prng_4d(A,X0,L)
    Z=(np.floor((2**32)*Y)).astype(np.uint32)
    Z_bytes = Z.view(np.uint8)
    psi = Z_bytes[:L].reshape((r,l))
    return psi

#shuffle/mask
def sequence_R(A2,X0,U,L):
    R = np.zeros(L,dtype=int)
    X = np.array(X0,dtype=float)
    for i in range(L):
        X = np.dot(A2,X)%1.0
        r=int(np.floor(X[1]*U))
        r=max(0,min(U-1,r))
        R[i]=r
    return R

def shuffle(Iblocks,X0,A):
    U=len(Iblocks)
    R=sequence_R(A["A2_primary"],X0["X0_primary"],U,U)
    Rt=sequence_R(A["A2_secondary"],X0["X0_secondary"],U,U)
    perm1 = Iblocks.copy()
    for i in range(U): perm1[i],perm1[R[i]] = perm1[R[i]].copy(),perm1[i].copy()
    perm2 = perm1.copy()
    for i in range(U): perm2[i],perm2[Rt[i]] = perm2[Rt[i]].copy(),perm2[i].copy()
    return perm2

def deshuffle(Cblocks,X0,A):
    U=len(Cblocks)
    R = sequence_R(A["A2_primary"],X0["X0_primary"],U,U)
    Rt=sequence_R(A["A2_secondary"],X0["X0_secondary"],U,U)
    perm1 = Cblocks.copy()
    for i in reversed(range(U)): perm1[i],perm1[Rt[i]] = perm1[Rt[i]].copy(),perm1[i].copy()
    perm0 = perm1.copy()
    for i in reversed(range(U)): perm0[i],perm0[R[i]] = perm0[R[i]].copy(),perm0[i].copy()
    return perm0

def mask(Iblocks, omega, psi):
    U = len(Iblocks)
    l = len(Iblocks[0])
    masked = []
    prev = np.zeros_like(Iblocks[0], dtype=np.uint8)

    for i in range(U):
        block = Iblocks[i].astype(np.uint8)
        T = ((omega[i % omega.shape[0]].astype(np.uint16)) ^(psi[i % psi.shape[0]].astype(np.uint16))) % 256
        shift = int(T[0] % l)
        shifted = np.roll(block, shift, axis=1).astype(np.uint16)
        prev_block = prev.astype(np.uint16)
        masked_block = (shifted ^ prev_block) % 256
        masked.append(masked_block.astype(np.uint8))
        prev = masked_block.copy()

    return np.array(masked)

def unmask(Cblocks, omega, psi):
    U = len(Cblocks)
    l = len(Cblocks[0])
    original = []
    prev = np.zeros_like(Cblocks[0], dtype=np.uint8)

    for i in range(U):
        C = Cblocks[i].astype(np.uint8)
        T = ((omega[i % omega.shape[0]].astype(np.uint16)) ^(psi[i % psi.shape[0]].astype(np.uint16))) % 256
        shift = int(T[0] % l)
        prev_block = prev.astype(np.uint16)
        shifted = (C.astype(np.uint16) ^ prev_block) % 256
        block = np.roll(shifted, -shift, axis=1)
        original.append(block.astype(np.uint8))
        prev = C.copy()

    return np.array(original)

#data

def prepare_image(im, l):
    h,w=im.shape[:2]
    h_new,h_mod = h-(h%l),h%l
    w_new,w_mod = w-(w%l),w%l
    im = im[:h_new,:w_new]
    blocks=[]
    for i in range(0,h_new,l):
        for j in range(0,w_new,l):
            blocks.append(im[i:i+l,j:j+l].copy())
    return np.array(blocks)

def save_image(blocks,l,original_shape,filename):
    h,w = original_shape
    h_new, w_new = h-(h%l), w-(w%l)
    out = np.zeros((h_new,w_new),dtype=np.uint8)
    idx=0
    for i in range(0,h_new,l):
        for j in range(0,w_new,l):
            out[i:i+l,j:j+l]=blocks[idx]
            idx+=1
    cv.imwrite(filename,out)

def get_data() -> Tuple[npt.NDArray, int, int]:
    im = cv.imread("Imagenes/decyphered_image.png", cv.IMREAD_GRAYSCALE)
    rounds = 3
    l = 16
    return im, rounds, l

#main

def encrypt_image(im,rounds,l,A,X0):
    Iblocks=prepare_image(im,l)
    U=len(Iblocks)
    n=l
    r=rounds
    omega = get_omega(A["A4"],X0["X0_A4"],l,n)
    psi = get_psi(A["A4"],X0["X0_A4"],r,l)

    for _ in range(rounds):
        Iblocks = shuffle(Iblocks,X0,A)
        Iblocks = mask(Iblocks,omega,psi)

    return Iblocks

def decrypt_image(Cblocks,rounds,l,A,X0):
    U=len(Cblocks)
    n=l
    r=rounds
    omega = get_omega(A["A4"],X0["X0_A4"],l,n)
    psi   = get_psi(A["A4"],X0["X0_A4"],r,l)

    for _ in range(rounds):
        Cblocks = unmask(Cblocks,omega,psi)
        Cblocks = deshuffle(Cblocks,X0,A)
    return Cblocks

def encode():
    im, rounds, l = get_data()
    original_shape = im.shape
    A = generate_A()
    X0 = generate_X0()
    save_keys(A,X0)
    Cblocks = encrypt_image(im,rounds,l,A,X0)
    save_image(Cblocks,l,original_shape,"encrypted.png")

def decode():
    C = cv.imread("encrypted.png",cv.IMREAD_GRAYSCALE)
    original_shape = C.shape
    l = 16
    rounds = 3
    A,X0 = load_keys()
    Cblocks = prepare_image(C,l)
    Iblocks = decrypt_image(Cblocks,rounds,l,A,X0)
    save_image(Iblocks,l,original_shape,"decrypted.png")

if __name__=="__main__":
    encode()
    decode()
