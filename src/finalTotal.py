import numpy as np
import numpy.typing as npt
import cv2 as cv
import json
from typing import Tuple

#Pres -> 8x8
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
    return np.array([[1,a],[b,a*b+1]], dtype=float)

def generate_params():
    return {
        "p_omega": np.random.randint(1,10,4).tolist(),
        "p_psi": np.random.randint(1,10,4).tolist(),
        "p_a2_primary": np.random.randint(1,10,2).tolist(),
        "p_a2_secondary": np.random.randint(1,10,2).tolist()
    }

def generate_X0():
    def rand_nonzero(n):
        x = np.random.random(n)
        while np.allclose(x,0): x=np.random.random(n)
        return x

    return {
        "X0_omega": rand_nonzero(4),
        "X0_psi": rand_nonzero(4),
        "X0_a2_primary": rand_nonzero(2),
        "X0_a2_secondary": rand_nonzero(2)
    }

def save_keys(params,X0):
    keys = {"params": params, "X0": {k:v.tolist() for k,v in X0.items()}}
    with open("keys.json","w") as f: json.dump(keys,f,indent=4)

def load_keys():
    with open("keys.json","r") as f:
        data=json.load(f)
    p=data["params"]
    X0={k:np.array(v) for k,v in data["X0"].items()}
    A={
        "A_omega":cat_map_4d_matrix(*p["p_omega"]),
        "A_psi":cat_map_4d_matrix(*p["p_psi"]),
        "A_a2_primary":cat_map_2d_matrix(*p["p_a2_primary"]),
        "A_a2_secondary":cat_map_2d_matrix(*p["p_a2_secondary"])
    }
    return A,X0

def prng_4d(A, X0, L):
    X=X0.copy()
    Y_list=[]
    U=np.array([2,3,5,1],dtype=float)
    iterations = L // 16 + 1
    for _ in range(iterations):
        t_vec=np.dot(U,X)
        t=int(np.ceil(t_vec))+1
        for _ in range(t):
            X=np.dot(A,X)%1.0
        Y_list.append(X.copy())
    Y=np.concatenate(Y_list)[:L]
    return Y

def get_omega(A,X0,l,n):
    L=l*n
    Y=prng_4d(A,X0,L)
    Z=(np.floor((2**32)*Y)).astype(np.uint32)
    Z_bytes = Z.view(np.uint8)
    return Z_bytes[:L].reshape((l,n))

def get_psi(A,X0,r,l):
    L=l*r
    Y=prng_4d(A,X0,L)
    Z=(np.floor((2**32)*Y)).astype(np.uint32)
    Z_bytes = Z.view(np.uint8)
    return Z_bytes[:L].reshape((r,l))

def sequence_R(A2,X0,U,L):
    R=np.zeros(L,dtype=int)
    X=np.array(X0)
    for i in range(L):
        X=np.dot(A2,X)%1.0
        r=int(np.floor(X[1]*U))
        r=max(0,min(U-1,r))
        R[i]=r
    return R

def shuffle(blocks,X0,A):
    U=len(blocks)
    R = sequence_R(A["A_a2_primary"],  X0["X0_a2_primary"], U,U)
    Rt= sequence_R(A["A_a2_secondary"],X0["X0_a2_secondary"],U,U)

    perm1 = blocks.copy()
    for i in range(U):
        perm1[i], perm1[R[i]] = perm1[R[i]].copy(), perm1[i].copy()

    perm2 = perm1.copy()
    for i in range(U):
        perm2[i], perm2[Rt[i]] = perm2[Rt[i]].copy(), perm2[i].copy()

    return perm2

def deshuffle(blocks,X0,A):
    U=len(blocks)
    R = sequence_R(A["A_a2_primary"],  X0["X0_a2_primary"], U,U)
    Rt= sequence_R(A["A_a2_secondary"],X0["X0_a2_secondary"],U,U)

    perm1=blocks.copy()
    for i in reversed(range(U)):
        perm1[i], perm1[Rt[i]] = perm1[Rt[i]].copy(), perm1[i].copy()

    perm0=perm1.copy()
    for i in reversed(range(U)):
        perm0[i], perm0[R[i]] = perm0[R[i]].copy(), perm0[i].copy()

    return perm0

def mask(blocks,omega,psi):
    U=len(blocks)
    l=len(blocks[0])
    out=[]
    prev=np.zeros_like(blocks[0],dtype=np.uint8)

    for i in range(U):
        block=blocks[i]
        T=((omega[i % omega.shape[0]].astype(np.uint16)) ^
           (psi[i % psi.shape[0]].astype(np.uint16))) %256

        shift=int(T[0] % l)
        shifted=np.roll(block, shift, axis=1).astype(np.uint16)
        masked=(shifted ^ prev)%256
        out.append(masked.astype(np.uint8))

        prev=masked.copy()
    return np.array(out)

def unmask(blocks,omega,psi):
    U=len(blocks)
    l=len(blocks[0])
    out=[]
    prev=np.zeros_like(blocks[0],dtype=np.uint8)

    for i in range(U):
        C=blocks[i]
        T=((omega[i % omega.shape[0]].astype(np.uint16)) ^
           (psi[i % psi.shape[0]].astype(np.uint16))) %256

        shift=int(T[0] % l)
        shifted=(C.astype(np.uint16) ^ prev.astype(np.uint16))%256
        block=np.roll(shifted,-shift,axis=1)

        out.append(block.astype(np.uint8))
        prev=C.copy()

    return np.array(out)

def prepare_image(im,l):
    h,w=im.shape
    h_new=h-(h%l)
    w_new=w-(w%l)
    blocks=[]
    for i in range(0,h_new,l):
        for j in range(0,w_new,l):
            blocks.append(im[i:i+l,j:j+l].copy())
    return np.array(blocks)

def blocks_to_image(blocks, original_shape, l):
    h,w=original_shape
    h_new=h-(h%l)
    w_new=w-(w%l)
    out=np.zeros((h_new,w_new),dtype=np.uint8)
    idx=0
    for i in range(0,h_new,l):
        for j in range(0,w_new,l):
            out[i:i+l,j:j+l]=blocks[idx]
            idx+=1
    return out

def save_image(blocks,l,original_shape,filename):
    out = blocks_to_image(blocks, original_shape,l)
    cv.imwrite(filename,out)

def save_stage_image(blocks, l, original_shape, filename):
    img = blocks_to_image(blocks, original_shape, l)
    cv.imwrite(filename, img)

def encrypt_image(im,rounds,l,A,X0):
    blocks=prepare_image(im,l)
    original_shape = im.shape

    omega=get_omega(A["A_omega"],X0["X0_omega"],l,l)
    psi  =get_psi( A["A_psi"], X0["X0_psi"], rounds,l)

    for r in range(rounds):
        #shuffle
        blocks=shuffle(blocks,X0,A)
        save_stage_image(blocks, l, original_shape, f"stage_shuffle_round{r+1}.png")

        #mask
        blocks=mask(blocks,omega,psi)
        save_stage_image(blocks, l, original_shape, f"stage_mask_round{r+1}.png")
    return blocks

def decrypt_image(blocks,rounds,l,original_shape,A,X0):
    omega=get_omega(A["A_omega"],X0["X0_omega"],l,l)
    psi  =get_psi( A["A_psi"], X0["X0_psi"], rounds,l)

    for r in range(rounds):
        #unmask
        blocks=unmask(blocks,omega,psi)
        save_stage_image(blocks, l, original_shape, f"stage_unmask_round{r+1}.png")

        #deshuffle
        blocks=deshuffle(blocks,X0,A)
        save_stage_image(blocks, l, original_shape, f"stage_deshuffle_round{r+1}.png")
    return blocks

def npcr_uaci(img1: np.ndarray, img2: np.ndarray) -> Tuple[float,float]:
    if img1.shape != img2.shape: 
        raise ValueError("Images must have same shape")

    diff = img1 != img2
    NPCR = diff.sum() * 100.0 / diff.size

    UACI = np.abs(img1.astype(np.int16) - img2.astype(np.int16)).mean() * 100.0 / 255.0
    return NPCR, UACI

def get_data():
    im = cv.imread("decypheredImage.png", cv.IMREAD_GRAYSCALE)
    print("Dimensiones de decypheredImage.png:", im.shape)
    return im, 3, 8

def encode():
    im, rounds, l = get_data()
    original_shape = im.shape
    np.save("shape.npy", np.array(original_shape))

    params=generate_params()
    X0=generate_X0()
    save_keys(params,X0)

    A={
        "A_omega":cat_map_4d_matrix(*params["p_omega"]),
        "A_psi":cat_map_4d_matrix(*params["p_psi"]),
        "A_a2_primary":cat_map_2d_matrix(*params["p_a2_primary"]),
        "A_a2_secondary":cat_map_2d_matrix(*params["p_a2_secondary"])
    }

    Cblocks=encrypt_image(im,rounds,l,A,X0)
    save_image(Cblocks,l,original_shape,"encrypted.png")

def decode():
    C = cv.imread("encrypted.png", cv.IMREAD_GRAYSCALE)
    original_shape = tuple(np.load("shape.npy"))
    l, rounds = 8, 3

    A,X0 = load_keys()

    Cblocks=prepare_image(C,l)
    Iblocks=decrypt_image(Cblocks,rounds,l,original_shape,A,X0)
    save_image(Iblocks,l,original_shape,"decrypted.png")

    O = cv.imread("decypheredImage.png", cv.IMREAD_GRAYSCALE)
    O_mod = O.copy()
    O_mod= O_mod.astype(np.uint16)
    O_mod[0,0] = (O_mod[0,0] + 1) % 256

    C_mod_blocks = encrypt_image(O_mod, rounds, l, A, X0)
    C_mod = blocks_to_image(C_mod_blocks, original_shape, l)

    I_img = blocks_to_image(Iblocks, original_shape, l)

    cv.imwrite("C1.png", C) 
    cv.imwrite("C2.png", C_mod)

    H = min(C.shape[0], C_mod.shape[0])
    W = min(C.shape[1], C_mod.shape[1])
    C     = C[:H, :W]
    C_mod = C_mod[:H, :W]

    diff = cv.absdiff(C, C_mod)

    diff_visual = np.clip(diff * 20, 0, 255).astype(np.uint8)
    cv.imwrite("Diff.png", diff_visual)

    diff_binary = (diff > 0).astype(np.uint8) * 255
    cv.imwrite("Diff_binary.png", diff_binary)

    NPCR, UACI = npcr_uaci(I_img, C_mod)
    print("\n========= NPCR & UACI =========")
    print(f"NPCR = {NPCR:.6f}%")
    print(f"UACI = {UACI:.6f}%")
    print("================================")

if __name__=="__main__":
    encode()
    decode()
