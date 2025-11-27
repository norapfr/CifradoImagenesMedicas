import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import sys
import os


from Scripts import encrypt_image, load_keys, save_image, prepare_image

# ======== MÉTRICAS ========
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
    """
    Genera y guarda histogramas de la imagen original y cifrada.
    
    Parámetros:
        original: np.array, imagen original
        encrypted: np.array, imagen cifrada
        title: str, título opcional para los gráficos
        save_path: str, ruta donde guardar la imagen (incluye nombre y extensión)
        show: bool, si True muestra la imagen
    """
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

# ======== CORTE DE IMÁGENES ========
def crop_to_blocks(im, l):
    """Recorta la imagen para que sea múltiplo de l x l"""
    h, w = im.shape[:2]
    h_new = h - (h % l)
    w_new = w - (w % l)
    return im[:h_new, :w_new]

# ======== ANÁLISIS ========
def analizar_imagen(path_original, path_cifrada, path_descifrada, nombre=""):

    print("\n===============================================")
    print(f"📌 ANALIZANDO: {nombre}")
    print("===============================================\n")

    # Cargar imágenes
    O = cv.imread(path_original, cv.IMREAD_GRAYSCALE)
    C = cv.imread(path_cifrada, cv.IMREAD_GRAYSCALE)
    D = cv.imread(path_descifrada, cv.IMREAD_GRAYSCALE)

    if O is None or C is None or D is None:
        print("❌ Error cargando una de las imágenes.")
        return

    # ---- Parámetros de tu cifrado ----
    rounds = 3
    l = 16
    A, X0 = load_keys()

    # Recortar todas las imágenes al mismo tamaño
    O = crop_to_blocks(O, l)
    D = crop_to_blocks(D, l)
    C = crop_to_blocks(C, l)

    # ---- Métricas de descifrado ----
    print("➡️ Métricas del descifrado:")
    print("PSNR:", safe_psnr(O, D))
    print("MSE :", mse(O, D))

    # ---- Correlación ----
    print("\n➡️ Correlación en imagen cifrada:")
    print("Correlación horizontal:", correlation_coefficient(C))

    # ---- NPCR & UACI ----
    print("\n➡️ NPCR & UACI:")

    # Crear una versión con 1 píxel modificado de forma segura
    O_mod = O.astype(np.uint16).copy()
    O_mod[0, 0] = (O_mod[0, 0] + 1) % 256
    O_mod = O_mod.astype(np.uint8)

    # Cifrar la imagen modificada
    C_mod_blocks = encrypt_image(O_mod, rounds, l, A, X0)

    # Reconstruir imagen cifrada modificada
    C_mod = np.zeros_like(C)
    idx = 0
    for i in range(0, C.shape[0], l):
        for j in range(0, C.shape[1], l):
            C_mod[i:i + l, j:j + l] = C_mod_blocks[idx]
            idx += 1

    NPCR, UACI = npcr_uaci(C, C_mod)
    print("NPCR:", NPCR)
    print("UACI:", UACI)

    # ---- Guardar histogramas ----
    hist_path = os.path.join("Resultados", f"histograma_{nombre.replace(' ', '_')}.png")
    print("\n📊 Mostrando y guardando histogramas...")
    plot_histograms(O, C, title=nombre, save_path=hist_path, show=True)


# ============================================
#           ANALIZAR 3 IMÁGENES
# ============================================


print("\n🚀 INICIANDO ANÁLISIS COMPLETO...")

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

print("\n🎉 Análisis finalizado.\n")

'''
PSNR (Peak Signal-to-Noise Ratio) y MSE (Mean Squared Error):
MSE: mide el error cuadrático medio entre la imagen original y la descifrada.
PSNR es una medida de calidad de la reconstrucción de la imagen: cuanto mayor, mejor.
    Valor 0.0 significa que la imagen descifrada es idéntica a la original.
    PSNR = inf (infinito) ocurre porque MSE = 0, lo que indica perfecta recuperación.
        Interpretación: Tu cifrado y descifrado funcionan sin pérdida de información.

Correlación horizontal en imagen cifrada: Mide la relación entre píxeles adyacentes en la imagen cifrada.
    Valores cercanos a 0 significan que no hay correlación, es decir, los píxeles son prácticamente aleatorios.
    Valores negativos muy bajos también son normales y muestran independencia de píxeles.
        Interpretación: La imagen cifrada no revela patrones, lo que es bueno para seguridad.

NPCR (Number of Pixels Change Rate) y UACI (Unified Average Changing Intensity):
Estas métricas se usan para analizar resistencia a cambios pequeños en la imagen original:
NPCR: porcentaje de píxeles que cambian en la imagen cifrada cuando se cambia 1 píxel en la original.
    ~100% significa que cualquier cambio mínimo produce cambios masivos en la imagen cifrada (deseable).
        MRI 1 y 2: NPCR ≈ 99.6 → excelente.
        MRI 3: NPCR ≈ 3.1 → malo, indica que el cifrado no está propagando bien los cambios.
UACI: indica la intensidad promedio de cambio relativo en los píxeles.
    Valores ~33% son típicos de un cifrado fuerte.
    MRI 1 y 2: UACI ≈ 33 → bien.
    MRI 3: UACI ≈ 0.01 → muy bajo → cifrado débil o incompleto.

Conclusión de tus resultados
MRI 1 y 2: cifrado sólido. La descifrado es perfecto, los píxeles cifrados son aleatorios y pequeños cambios en la original afectan toda la imagen cifrada.
MRI 3: algo está fallando. El cifrado no genera suficiente aleatoriedad (NPCR y UACI muy bajos). Esto podría deberse a:
    Tamaño de la imagen no múltiplo de l=16, truncamiento de bloques.
    Problemas al generar claves A y X0.
    Error al preparar o reconstruir la imagen.


Histogramas:
Izquierda: Histograma Original
El eje X representa los niveles de gris (0–255).
El eje Y representa la cantidad de píxeles para cada nivel de gris.
Observas que hay muchos píxeles en los niveles bajos de gris, típico en imágenes médicas como MRI, donde hay mucho fondo oscuro y zonas específicas de intensidad.

Derecha: Histograma Cifrado
Muestra la distribución de niveles de gris después de aplicar tu cifrado.
Ahora todos los niveles de gris están aproximadamente igual distribuídos.
Esto significa que la imagen cifrada no tiene patrones visibles y es prácticamente aleatoria.
Este comportamiento es deseable en cifrado de imágenes: evita que alguien pueda inferir información visual de la imagen cifrada.
'''


'''
===============================================
📌 ANALIZANDO: MRI 1
===============================================

➡️ Métricas del descifrado:
PSNR: inf
MSE : 0.0

➡️ Correlación en imagen cifrada:
Correlación horizontal: -0.005492672331771436

➡️ NPCR & UACI:
NPCR: 99.62488185255198
UACI: 33.503807

📊 Mostrando histogramas...

===============================================
📌 ANALIZANDO: MRI 2
===============================================

➡️ Métricas del descifrado:
PSNR: inf
MSE : 0.0

➡️ Correlación en imagen cifrada:
Correlación horizontal: 0.0022035103425170704

➡️ NPCR & UACI:
NPCR: 99.6234170751634
UACI: 33.506397

📊 Mostrando histogramas...

===============================================
📌 ANALIZANDO: MRI 3
===============================================

➡️ Métricas del descifrado:
PSNR: inf
MSE : 0.0

➡️ Correlación en imagen cifrada:
Correlación horizontal: 0.0004065524725433982

➡️ NPCR & UACI:
NPCR: 3.1258468834688347
UACI: 0.012258223

📊 Mostrando histogramas...

🎉 Análisis finalizado.

'''