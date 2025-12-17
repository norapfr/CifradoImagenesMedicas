import sys
import shutil
from pathlib import Path
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr


BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"         # core, encoder, decoder
SCRIPTS_DIR = BASE_DIR / "Scripts" # main.py
KEYS_DIR = BASE_DIR / "Claves"
RESULTS_DIR = BASE_DIR / "Resultados"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))


from core import *
from encoder import encode
from decoder import decode
from main import load_keys

# ==========================================================
# MÉTRICAS
# ==========================================================
def safe_psnr(original, decrypted):
    mse_val = np.mean((original.astype(np.float32) - decrypted.astype(np.float32)) ** 2)
    if mse_val == 0:
        return float("inf")
    return psnr(original, decrypted)

def mse(original, decrypted):
    return np.mean((original.astype(np.float32) - decrypted.astype(np.float32)) ** 2)

def correlation_coefficient(image):
    x = image[:, 1:].ravel()
    y = image[:, :-1].ravel()
    return np.corrcoef(x, y)[0, 1]

# ==========================================================
# HISTOGRAMAS
# ==========================================================
def plot_histograms(original, encrypted, title="", save_path=None, show=True):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(original.ravel(), bins=256)
    plt.title(f"Histograma Original {title}")
    plt.subplot(1, 2, 2)
    plt.hist(encrypted.ravel(), bins=256)
    plt.title(f"Histograma Cifrada {title}")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f" Histograma guardado en: {save_path}")
    if show:
        plt.show()
    plt.close()


def crop_to_blocks(im, l):
    h, w = im.shape
    return im[: h - (h % l), : w - (w % l)]

def activar_key(path_key):
    if not path_key.exists():
        raise FileNotFoundError(f"No existe la key: {path_key}")
    shutil.copy(path_key, BASE_DIR / "keys.json")
    print(f"🔐 Key activada: {path_key.name}")

# ==========================================================
# ANÁLISIS DE UNA IMAGEN
# ==========================================================
def analizar_imagen(path_original, path_cifrada, path_descifrada, path_key, nombre=""):

    print("\n===============================================")
    print(f"ANALIZANDO: {nombre}")
    print("===============================================\n")

    activar_key(path_key)

   
    O = cv.imread(str(path_original), cv.IMREAD_GRAYSCALE)
    C = cv.imread(str(path_cifrada), cv.IMREAD_GRAYSCALE)
    D = cv.imread(str(path_descifrada), cv.IMREAD_GRAYSCALE)
    if O is None or C is None or D is None:
        print(" Error cargando imágenes")
        return

    rounds = 3
    l = 16

    A, X0 = load_keys()

    O = crop_to_blocks(O, l)
    C = crop_to_blocks(C, l)
    D = crop_to_blocks(D, l)

    
    print("Métricas del descifrado:")
    print("PSNR:", safe_psnr(O, D))
    print("MSE :", mse(O, D))

    
    print("\nCorrelación en imagen cifrada:")
    print("Correlación horizontal:", correlation_coefficient(C))

   
    O_mod = O.copy().astype(np.uint16)
    O_mod[0,0] = (O_mod[0,0] + 1) % 256
    O_mod = O_mod.astype(np.uint8)

    C_mod_blocks = encrypt_image(O_mod, rounds, l, A, X0)
    C_mod = blocks_to_image(C_mod_blocks, O.shape, l)

    H = min(C.shape[0], C_mod.shape[0])
    W = min(C.shape[1], C_mod.shape[1])
    C = C[:H, :W]
    C_mod = C_mod[:H, :W]

    diff = cv.absdiff(C, C_mod)
    diff_visual = np.clip(diff * 20, 0, 255).astype(np.uint8)
    diff_binary = (diff > 0).astype(np.uint8) * 255

    cv.imwrite(f"{RESULTS_DIR}/C1_{nombre}.png", C)
    cv.imwrite(f"{RESULTS_DIR}/C2_{nombre}.png", C_mod)
    cv.imwrite(f"{RESULTS_DIR}/Diff_{nombre}.png", diff_visual)
    cv.imwrite(f"{RESULTS_DIR}/Diff_binary_{nombre}.png", diff_binary)

    Cblocks = prepare_image(C, l)
    Iblocks = decrypt_image(Cblocks, rounds, l, O.shape, A, X0)
    I_img = blocks_to_image(Iblocks, O.shape, l)

    NPCR, UACI = npcr_uaci(I_img, C_mod)

    print("\n========= NPCR & UACI =========")
    print(f"NPCR = {NPCR:.6f}%")
    print(f"UACI = {UACI:.6f}%")
    print("================================")

    # Histogramas
    hist_path = RESULTS_DIR / f"histograma_{nombre}.png"
    plot_histograms(O, C, title=nombre, save_path=str(hist_path), show=True)

# ==========================================================
# EJECUCIÓN DEL TEST
# ==========================================================
if __name__ == "__main__":

    print("\nINICIANDO ANÁLISIS DIFERENCIAL COMPLETO...\n")

    analizar_imagen(
        BASE_DIR / "Imagenes/mri1.jpg",
        BASE_DIR / "Imagenes/MR1_cifrada.png",
        BASE_DIR / "Imagenes/MR1_descifrada.png",
        KEYS_DIR / "keys_mr1.json",
        nombre="MRI_1"
    )

    analizar_imagen(
        BASE_DIR / "Imagenes/mri2.jpg",
        BASE_DIR / "Imagenes/MR2_cifrada.png",
        BASE_DIR / "Imagenes/MR2_descifrada.png",
        KEYS_DIR / "keys_mr2.json",
        nombre="MRI_2"
    )

    analizar_imagen(
        BASE_DIR / "Imagenes/mri3.jpg",
        BASE_DIR / "Imagenes/MR3_cifrada.png",
        BASE_DIR / "Imagenes/MR3_descifrada.png",
        KEYS_DIR / "keys_mr3.json",
        nombre="MRI_3"
    )

    print("\nANÁLISIS DIFERENCIAL FINALIZADO.\n")
