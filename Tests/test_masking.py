import sys
from pathlib import Path
import numpy as np
import cv2 as cv

sys.path.append(str(Path(__file__).parent.parent / "scr"))
from core import generate_A, generate_X0, get_omega, get_psi, prepare_image, mask

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

def test_mask_effect(l: int = 16, rounds: int = 3, image_name: str = "mri1.jpg"):
    img_path = Path("Imagenes") / image_name
    if not img_path.exists():
        raise FileNotFoundError(f"No se encontró {img_path}.")
    im = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    if im is None:
        raise RuntimeError("Error cargando la imagen")
    
    original_shape = im.shape
    Iblocks = prepare_image(im, l)
    A = generate_A()
    X0 = generate_X0()
    omega = get_omega(A["A4"], X0["X0_A4"], l, l)
    psi = get_psi(A["A4"], X0["X0_A4"], rounds, l)
    masked_blocks = mask(Iblocks.copy(), omega, psi)
    masked_img = blocks_to_image(masked_blocks, l, original_shape)
    output_dir = Path("Tests/images")
    output_dir.mkdir(exist_ok=True)
    out_name = output_dir / f"masked_{image_name.split('.')[0]}.png"
    cv.imwrite(str(out_name), masked_img)
    print(f"Generado: {out_name}")


if __name__ == "__main__":
    test_mask_effect(image_name="mri1.jpg")
