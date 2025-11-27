import sys
from pathlib import Path
import numpy as np
import cv2 as cv

sys.path.append(str(Path(__file__).parent.parent / "Scripts"))
from main import generate_A, generate_X0, prepare_image, shuffle

# convertir de bloques a imagen para poder mostrar y guardar el resultado
def blocks_to_image(blocks: np.ndarray, l: int, original_shape: tuple[int, int]) -> np.ndarray:
    h, w = original_shape
    h_new, w_new = h - (h % l), w - (w % l)
    out = np.zeros((h_new, w_new), dtype=np.uint8)
    idx = 0
    for i in range(0, h_new, l):
        for j in range(0, w_new, l):
            out[i:i+l, j:j+l] = blocks[idx]
            idx += 1
    return out


def test_shuffle_effect(l: int = 16, image_name: str = "mri1.jpg"):
    # Cargar imagen
    img_path = Path("Imagenes") / image_name
    if not img_path.exists():
        raise FileNotFoundError(f"No se encontró {img_path}.")
    im = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError("Error cargando la imagen")
    
    original_shape = im.shape
    Iblocks = prepare_image(im, l)

    # Generar claves
    A = generate_A()
    X0 = generate_X0()

    # Aplicar shuffle
    shuffled_blocks = shuffle(Iblocks.copy(), X0, A)
    shuffled_img = blocks_to_image(shuffled_blocks, l, original_shape)
    
    # Guardar resultado
    output_dir = Path("Tests/images")
    output_dir.mkdir(exist_ok=True)
    out_name = output_dir / f"shuffled_{image_name.split('.')[0]}.png"
    cv.imwrite(str(out_name), shuffled_img)
    print(f"Generado: {out_name}")


if __name__ == "__main__":
    test_shuffle_effect(image_name="mri1.jpg")
