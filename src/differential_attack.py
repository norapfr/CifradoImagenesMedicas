from core import *
from encoder import encode
from decoder import decode

def differential_attack():
    encode()
    decode()

    C = cv.imread("encrypted.png", cv.IMREAD_GRAYSCALE)
    original_shape = C.shape
    l, rounds = 1, 3

    A, X0 = load_keys()

    O = cv.imread("decypheredImage.png", cv.IMREAD_GRAYSCALE)
    O_mod = O.copy().astype(np.uint16)
    O_mod[0,0] = (O_mod[0,0] + 1) % 256

    C_mod_blocks = encrypt_image(O_mod, rounds, l, A, X0)
    C_mod = blocks_to_image(C_mod_blocks, original_shape, l)

    H = min(C.shape[0], C_mod.shape[0])
    W = min(C.shape[1], C_mod.shape[1])
    C = C[:H, :W]
    C_mod = C_mod[:H, :W]

    diff = cv.absdiff(C, C_mod)
    diff_visual = np.clip(diff * 20, 0, 255).astype(np.uint8)
    diff_binary = (diff > 0).astype(np.uint8) * 255

    cv.imwrite("C1.png", C)
    cv.imwrite("C2.png", C_mod)
    cv.imwrite("Diff.png", diff_visual)
    cv.imwrite("Diff_binary.png", diff_binary)

    Cblocks = prepare_image(C, l)
    Iblocks = decrypt_image(Cblocks, rounds, l, original_shape, A, X0)
    I_img = blocks_to_image(Iblocks, original_shape, l)

    NPCR, UACI = npcr_uaci(I_img, C_mod)

    print("\n========= NPCR & UACI =========")
    print(f"NPCR = {NPCR:.6f}%")
    print(f"UACI = {UACI:.6f}%")
    print("================================")
