import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import sys
import os
from pathlib import Path

#para ejecutarlo -> python -m Tests.analisis_descifrado
sys.path.append(str(Path(__file__).parent.parent / "Scripts"))
from main import encrypt_image, load_keys, save_image, prepare_image

#méticas
def safe_psnr(original, decrypted):
    mse_val = np.mean((original.astype(np.float32) - decrypted.astype(np.float32))**2)
    if mse_val == 0:
        return float('inf')
    return psnr(original, decrypted)

def mse(original, decrypted):
    return np.mean((original.astype(np.float32) - decrypted.astype(np.float32)) ** 2)

def correlation_coefficient(image):
    x = image[:, 1:].ravel()
    y = image[:, :-1].ravel()
    return np.corrcoef(x, y)[0, 1]

def npcr_uaci(C1, C2):
    diff = C1 != C2
    NPCR = np.sum(diff) * 100 / diff.size
    UACI = np.sum(np.abs(C1.astype(np.float32) - C2.astype(np.float32))) * 100 / (255 * diff.size)
    return NPCR, UACI

def plot_histograms(original, encrypted, title="", save_path=None, show=True):

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(original.ravel(), bins=256)
    plt.title(f"Histograma Original {title}")

    plt.subplot(1, 2, 2)
    plt.hist(encrypted.ravel(), bins=256)
    plt.title(f"Histograma Cifrada {title}")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✅ Histograma guardado en: {save_path}")
    
    if show:
        plt.show()
    
    plt.close()

#corte de imágenes
def crop_to_blocks(im, l):
    h, w = im.shape[:2]
    h_new = h - (h % l)
    w_new = w - (w % l)
    return im[:h_new, :w_new]

#análisis
def analizar_imagen(path_original, path_cifrada, path_descifrada, nombre=""):

    print("\n===============================================")
    print(f"ANALIZANDO: {nombre}")
    print("===============================================\n")

    O = cv.imread(path_original, cv.IMREAD_GRAYSCALE)
    C = cv.imread(path_cifrada, cv.IMREAD_GRAYSCALE)
    D = cv.imread(path_descifrada, cv.IMREAD_GRAYSCALE)

    if O is None or C is None or D is None:
        print("Error cargando una de las imágenes.")
        return

    rounds = 3
    l = 16
    A, X0 = load_keys()
    O = crop_to_blocks(O, l)
    D = crop_to_blocks(D, l)
    C = crop_to_blocks(C, l)

    print("Métricas del descifrado:")
    print("PSNR:", safe_psnr(O, D))
    print("MSE :", mse(O, D))

    #correlación
    print("\nCorrelación en imagen cifrada:")
    print("Correlación horizontal:", correlation_coefficient(C))

    #NPCR & UACI
    print("\nNPCR & UACI:")

    O_mod = O.astype(np.uint16).copy()
    O_mod[0, 0] = (O_mod[0, 0] + 1) % 256
    O_mod = O_mod.astype(np.uint8)
    C_mod_blocks = encrypt_image(O_mod, rounds, l, A, X0)
    C_mod = np.zeros_like(C)
    idx = 0
    for i in range(0, C.shape[0], l):
        for j in range(0, C.shape[1], l):
            C_mod[i:i + l, j:j + l] = C_mod_blocks[idx]
            idx += 1

    NPCR, UACI = npcr_uaci(C, C_mod)
    print("NPCR:", NPCR)
    print("UACI:", UACI)

    #histogramas
    hist_path = os.path.join("Resultados", f"histograma_{nombre.replace(' ', '_')}.png")
    print("\nMostrando y guardando histogramas...")
    plot_histograms(O, C, title=nombre, save_path=hist_path, show=True)


#analizando 3 imágenes
print("\nINICIANDO ANÁLISIS DE 3 IMÁGENES...")

analizar_imagen(
    "Imagenes/mri1.jpg",
    "encrypted1.png",
    "decrypted1.png",
    nombre="MRI 1"
)

analizar_imagen(
    "Imagenes/mri2.jpg",
    "encrypted2.png",
    "decrypted2.png",
    nombre="MRI 2"
)

analizar_imagen(
    "Imagenes/mri3.jpg",
    "encrypted3.png",
    "decrypted3.png",
    nombre="MRI 3"
)

print("\nAnálisis finalizado.\n")

#análisis obtenidos:

'''
===============================================
ANALIZANDO: MRI 1
===============================================

Métricas del descifrado:
PSNR: inf
MSE : 0.0

Correlación en imagen cifrada:
Correlación horizontal: -0.005492672331771436

NPCR & UACI:
NPCR: 99.62488185255198
UACI: 33.503807

Mostrando histogramas...

===============================================
ANALIZANDO: MRI 2
===============================================

Métricas del descifrado:
PSNR: inf
MSE : 0.0

Correlación en imagen cifrada:
Correlación horizontal: 0.0022035103425170704

NPCR & UACI:
NPCR: 99.6234170751634
UACI: 33.506397

Mostrando histogramas...

===============================================
ANALIZANDO: MRI 3
===============================================

Métricas del descifrado:
PSNR: inf
MSE : 0.0

Correlación en imagen cifrada:
Correlación horizontal: 0.0004065524725433982

NPCR & UACI:
NPCR: 3.1258468834688347
UACI: 0.012258223

Mostrando histogramas...

Análisis finalizado.
'''