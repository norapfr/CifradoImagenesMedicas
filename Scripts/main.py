import numpy as np
import numpy.typing as npt
import cv2 as cv
import PIL as pillow
from typing import Tuple
import json
#l: tamaño de bloques (filas de omega)
#r: rondas
#n: cantidad de bloques de bytes en cada I (columnas de omega)
#I: sub-imagen
#U: número bloques

def select_ROI(im: np.ndarray, s: int = 16, tau: float = 5000) -> np.ndarray:

    # Convertir a gris si la imagen es color
    if len(im.shape) == 3:
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    else:
        gray = im.copy()

    H, W = gray.shape
    Hn = H - (H % s)
    Wn = W - (W % s)
    gray = gray[:Hn, :Wn]
    mask = np.zeros((Hn, Wn), dtype=np.uint8)
    for i in range(0, Hn, s):
        for j in range(0, Wn, s):

            block = gray[i:i+s, j:j+s].astype(np.float32)
            c_bar = np.mean(block)
            e = np.sum(np.abs(block - c_bar))
            if e > tau:
                mask[i:i+s, j:j+s] = 1 
            else:
                mask[i:i+s, j:j+s] = 0

    return mask


def get_chaotic_map() -> npt.NDArray[npt.NDArray]:
    
    chaotic_map = np.array(np.array([0]))#TODO

    return chaotic_map

#chaotic maps

#[2,1,2,2]
def cat_map_4d_matrix(a:int, b:int, c:int, d:int) -> npt.NDArray[npt.NDArray]:
    
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
    [A41,A42,A43,A44]],
    dtype=float)
    return A

def cat_map_2d_matrix(a:int, b: int)-> npt.NDArray[npt.NDArray]:
    A11=1
    A12=a
    A21=b
    A22=a*b+1
    A=np.array([
    [A11,A12],
    [A21,A22]],
    dtype=float)
    return A

#keys

def generate_A() -> dict({str:npt.NDArray[npt.NDArray]}):

    #shuffling
    a1 = np.random.randint(1, 10)
    b1 = np.random.randint(1, 10)
    a2 = np.random.randint(1, 10)
    b2 = np.random.randint(1, 10)
    
    info={"A4": cat_map_4d_matrix(2,1,2,2),
          "A2_primary": cat_map_2d_matrix(a1, b1),
          "A2_secondary": cat_map_2d_matrix(a2, b2)}

    return info

def generate_X0() -> dict({str:npt.NDArray[float]}):
    #omega/psi
    X0_A4=np.random.random(4)
    while np.allclose(X0_A4,0):
        X0_A4=np.random.random(4)

    #suffling
    X0_primary=np.random.random(2)
    while np.allclose(X0_primary,0):
        X0_primary=np.random.random(2)
    X0_secondary=np.random.random(2)
    while np.allclose(X0_secondary,0):
        X0_secondary=np.random.random(2)
    
    info={"X0_A4": X0_A4,
          "X0_primary": X0_primary,
          "X0_secondary": X0_secondary}

    return info

def save_keys(A, X0):
    keys = {
        "A": {k: v.tolist() for k, v in A.items()},
        "X0": {k: v.tolist() for k, v in X0.items()}
    }
    with open("keys.json", "w") as f:
        json.dump(keys, f, indent=4)


def load_keys():
    with open("keys.json", "r") as f:
        data = json.load(f)
    A  = {k: np.array(v) for k, v in data["A"].items()}
    X0 = {k: np.array(v) for k, v in data["X0"].items()}
    return A, X0

def prng_4d(A:npt.NDArray, X0:npt.NDArray, L:int) -> npt.NDArray:
    X = X0.copy()
    Y_list = []

    U = np.array([2, 3, 5, 1], dtype=float)
    iterations = L // 16 + 1 

    for _ in range(iterations):
        t_vec = np.dot(U, X)
        t = int(np.ceil(t_vec)) + 1
        for _ in range(t):
            X = np.dot(A, X) % 1.0
        Y_list.append(X.copy())

    Y = np.concatenate(Y_list)[:L]
    return Y

#bloques de lxl ~16/32
def prepare_image(im: npt.NDArray[npt.NDArray], l:int) -> npt.NDArray[npt.NDArray]:
    h, w = im.shape[:2]
    h_new = h - (h % l)
    w_new = w - (w % l)
    #print(h_new)
    #960
    #print(w_new)
    #1968
    im = im[:h_new, :w_new]

    Iblocks = []
    for i in range(0, h_new, l):
        for j in range(0, w_new, l):
            block = im[i:i+l, j:j+l].copy()
            Iblocks.append(block)

    #print(np.array(Iblocks))
    return np.array(Iblocks)

def get_omega(A:npt.NDArray[npt.NDArray], X0:npt.NDArray[int],l:int, n:int) -> npt.NDArray[npt.NDArray]:

    L=l*n
    Y = prng_4d(A, X0, L)
    Z = np.floor((2**32) * Y).astype(np.uint32)
    bin_str = "".join(f"{z:032b}" for z in Z)
    bytes_arr = [int(bin_str[i:i+8], 2) for i in range(0, len(bin_str), 8)]
    bytes_arr = bytes_arr[:L]
    omega = np.array(bytes_arr, dtype=np.uint8).reshape((l, n))
    return omega

def get_psi(A:npt.NDArray[npt.NDArray], X0:npt.NDArray[int],r:int, l:int) -> npt.NDArray[npt.NDArray]:

    L=l*r
    Y = prng_4d(A, X0, L)
    Z = np.floor((2**32) * Y).astype(np.uint32)
    bin_str = "".join(f"{z:032b}" for z in Z)
    bytes_arr = [int(bin_str[i:i+8], 2) for i in range(0, len(bin_str), 8)]
    bytes_arr = bytes_arr[:L]
    psi = np.array(bytes_arr, dtype=np.uint8).reshape((r, l))
    
    return psi

#U=numero de bloques
#L=longitud de la secuencia
#permutar todos los bloques de I? L=U
def sequence_R(A2,X0, U, L):

    R = np.zeros(L, dtype=int)
    X = np.array(X0, dtype=float)

    for i in range(L):
        X = np.dot(A2, X) % 1.0
        r = int(np.floor(X[1] * U)) + 1
        r = max(1, min(U, r))
        R[i] = r - 1
    return R

def shuffle(Iblocks, dict1: dict({str:npt.NDArray[float]}), dict2:dict({str:npt.NDArray[npt.NDArray]})):
    A2_primary=dict2["A2_primary"]
    A2_secondary=dict2["A2_secondary"]
    X0_primary=dict1["X0_primary"]
    X0_secondary=dict1["X0_secondary"]

    U=len(Iblocks)
    R=sequence_R(A2_primary, X0_primary, U, U)
    #print(R)
    Rt=sequence_R(A2_secondary, X0_secondary, U, U)
    #print(Rt)

    #columna
    perm1 = Iblocks.copy()

    for i in range(U):
        j = R[i]
        perm1[i],perm1[j]=perm1[j],perm1[i]

    #fila
    perm2 = perm1.copy()

    for i in range(U):
        j = Rt[i]
        perm2[i],perm2[j]=perm2[j],perm2[i]

    return perm2

def deshuffle(Cblocks, dict1, dict2):
    A2_primary   = dict2["A2_primary"]
    A2_secondary = dict2["A2_secondary"]
    X0_primary   = dict1["X0_primary"]
    X0_secondary = dict1["X0_secondary"]

    U = len(Cblocks)

    R  = sequence_R(A2_primary,   X0_primary,   U, U)
    Rt = sequence_R(A2_secondary, X0_secondary, U, U)

    #fila
    perm1 = Cblocks.copy()
    for i in reversed(range(U)):
        j = Rt[i]
        perm1[i], perm1[j] = perm1[j], perm1[i]

    #columna
    perm0 = perm1.copy()
    for i in reversed(range(U)):
        j = R[i]
        perm0[i], perm0[j] = perm0[j], perm0[i]

    return perm0

def mask(Iblocks, omega: npt.NDArray[np.uint8], psi: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:

    U = len(Iblocks)
    l = len(Iblocks[0])

    masked_blocks = []
    prev = np.zeros_like(Iblocks[0], dtype=np.uint8)

    for i in range(U):
        block = Iblocks[i].astype(np.uint8)
        #fila
        omega_block = omega[i % omega.shape[0]]
        psi_block   = psi[i % psi.shape[0]]
        #problema aquí
        #print(omega_block ^ psi_block)
        T = (omega_block ^ psi_block) % 256
        shift = int(T[0] % l)
        shifted_block = np.roll(block, shift, axis=1)
        masked_block = (shifted_block ^ prev) % 256
        masked_blocks.append(masked_block.astype(np.uint8))
        prev = masked_block.copy()

    return np.array(masked_blocks)

def unmask(Cblocks, omega, psi):
    U = len(Cblocks)
    l = len(Cblocks[0])
    
    original_blocks = []
    prev = np.zeros_like(Cblocks[0], dtype=np.uint8)

    for i in range(U):
        C = Cblocks[i].astype(np.uint8)
        omega_block = omega[i % omega.shape[0]]
        psi_block   = psi[i % psi.shape[0]]
        T = (omega_block ^ psi_block) % 256
        shift = int(T[0] % l)
        shifted_block = (C ^ prev) % 256
        block = np.roll(shifted_block, -shift, axis=1)
        original_blocks.append(block.astype(np.uint8))
        prev = C.copy()

    return np.array(original_blocks)

def encrypt_image(im: npt.NDArray[npt.NDArray],rounds: int,l: int,A: dict,X0: dict) -> npt.NDArray[npt.NDArray]:

    Iblocks = prepare_image(im, l)  
    U = len(Iblocks)
    n = l
    r = rounds

    omega = get_omega(A["A4"], X0["X0_A4"], l, n)
    psi   = get_psi(A["A4"], X0["X0_A4"], r, l)

    #print("omega")
    #print(omega)
    #print("psi")
    #print(psi)

    for _ in range(rounds):

        Iblocks = shuffle(Iblocks, X0, A)
        Iblocks = mask(Iblocks, omega, psi)
        omega = get_omega(A["A4"], X0["X0_A4"], l, n)
        psi = get_psi(A["A4"], X0["X0_A4"], r, l)

    return Iblocks

def decrypt_image(Cblocks, rounds, l, A, X0):
    
    U = len(Cblocks)
    n = l
    r = rounds

    omega = get_omega(A["A4"], X0["X0_A4"], l, n)
    psi   = get_psi(A["A4"], X0["X0_A4"], r, l)

    for _ in range(rounds):
        omega = get_omega(A["A4"], X0["X0_A4"], l, n)
        psi   = get_psi(A["A4"], X0["X0_A4"], r, l)
        Cblocks = unmask(Cblocks, omega, psi)
        Cblocks = deshuffle(Cblocks, X0, A)

    return Cblocks

def get_data() -> Tuple[npt.NDArray[npt.NDArray], int, int]:
    
    im = cv.imread("decypheredImage.png", cv.IMREAD_GRAYSCALE)
    #cv.IMREAD_GRAYSCALE
    rounds = 3
    l = 16

    return im, rounds, l


def save_image(blocks: npt.NDArray[npt.NDArray], l: int, original_shape: Tuple[int,int]):

    h, w = original_shape
    h_new = h - (h % l)
    w_new = w - (w % l)

    out = np.zeros((h_new, w_new), dtype=np.uint8)

    idx = 0
    for i in range(0, h_new, l):
        for j in range(0, w_new, l):
            out[i:i+l, j:j+l] = blocks[idx]
            idx += 1

    cv.imwrite("encrypted.png", out)

def encode() -> None:
    
    im, rounds, l = get_data()
    original_shape = im.shape
    #print(original_shape)
    #(963,1973,3)
    A  = generate_A()
    #print(A)
    X0 = generate_X0()
    #print(X0)
    #hasta aqui
    save_keys(A, X0)
    Cblocks = encrypt_image(im, rounds, l, A, X0)
    save_image(Cblocks, l, original_shape)

def decode() -> None:

    C = cv.imread("encrypted.png", cv.IMREAD_GRAYSCALE)
    original_shape = C.shape
    l = 16
    rounds = 3
    A, X0 = load_keys()
    Cblocks = prepare_image(C, l)
    Iblocks = decrypt_image(Cblocks, rounds, l, A, X0)
    save_image(Iblocks, l, original_shape)


if __name__=="__main__":
    encode()
    #get_data()
    #print(cat_map_4d_matrix(2,1,2,2))
