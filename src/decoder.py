from core import *

def decode():
    C = cv.imread("encrypted.png", cv.IMREAD_GRAYSCALE)
    original_shape = tuple(np.load("shape.npy"))
    l, rounds = 16, 3

    A, X0 = load_keys()

    Cblocks = prepare_image(C, l)
    Iblocks = decrypt_image(Cblocks, rounds, l, original_shape, A, X0)
    save_image(Iblocks, l, original_shape, "decrypted.png")
