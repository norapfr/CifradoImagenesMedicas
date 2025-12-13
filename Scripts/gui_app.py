import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv
import hashlib

# ==============================================================================
# 1. LÓGICA CRIPTOGRÁFICA (Matemáticas del caos)
# ==============================================================================

def cat_map_4d_matrix(a, b, c, d):
    E = 2*(a + 1)*(d + b*c*(d + 1)) + a*(c + 1)*(d + 1)
    F = a*(c + 3) + 2*b*c*(a + 1) + 2
    return np.array([
        [(2*b + c)*(d + 1) + 3*d + 1, 2*(b + 1), 2*b*c + c + 3, 4*b + 3],
        [E, 2*(a + b + a*b) + 1, F, 3*a + 4*b*(a + 1) + 1],
        [3*b*c*(d + 1) + 3*d, 3*b + 1, 3*b*c + 3, 6*b + 1],
        [c*(b + 1)*(d + 1) + d, b + 1, b*c + c + 1, 2*b + 2]
    ], dtype=float)

def cat_map_2d_matrix(a, b):
    return np.array([[1, a], [b, a*b+1]], dtype=float)

def get_deterministic_keys(seed_text):
    hash_object = hashlib.sha256(seed_text.encode())
    seed_int = int.from_bytes(hash_object.digest()[:4], 'big')
    rng = np.random.RandomState(seed_int)
    
    params = {
        "p_omega": rng.randint(1, 10, 4).tolist(),
        "p_psi": rng.randint(1, 10, 4).tolist(),
        "p_a2_primary": rng.randint(1, 10, 2).tolist(),
        "p_a2_secondary": rng.randint(1, 10, 2).tolist()
    }
    def rand_nonzero(n):
        x = rng.random_sample(n)
        while np.allclose(x, 0): x = rng.random_sample(n)
        return x
    X0 = {
        "X0_omega": rand_nonzero(4),
        "X0_psi": rand_nonzero(4),
        "X0_a2_primary": rand_nonzero(2),
        "X0_a2_secondary": rand_nonzero(2)
    }
    A = {
        "A_omega": cat_map_4d_matrix(*params["p_omega"]),
        "A_psi": cat_map_4d_matrix(*params["p_psi"]),
        "A_a2_primary": cat_map_2d_matrix(*params["p_a2_primary"]),
        "A_a2_secondary": cat_map_2d_matrix(*params["p_a2_secondary"])
    }
    return A, X0

def prng_4d(A, X0, L):
    X = X0.copy()
    Y_list = []
    U = np.array([2, 3, 5, 1], dtype=float)
    iterations = L // 16 + 1
    for _ in range(iterations):
        t = int(np.ceil(np.dot(U, X))) + 1
        for _ in range(t): X = np.dot(A, X) % 1.0
        Y_list.append(X.copy())
    return np.concatenate(Y_list)[:L]

def get_omega(A, X0, l, n):
    L = l * n
    Y = prng_4d(A, X0, L)
    Z = (np.floor((2**32) * Y)).astype(np.uint32)
    return Z.view(np.uint8)[:L].reshape((l, n))

def get_psi(A, X0, r, l):
    L = l * r
    Y = prng_4d(A, X0, L)
    Z = (np.floor((2**32) * Y)).astype(np.uint32)
    return Z.view(np.uint8)[:L].reshape((r, l))

def sequence_R(A2, X0, U, L):
    R = np.zeros(L, dtype=int)
    X = np.array(X0)
    for i in range(L):
        X = np.dot(A2, X) % 1.0
        r_val = int(np.floor(X[1] * U))
        R[i] = max(0, min(U - 1, r_val))
    return R

def shuffle(blocks, X0, A, mode='encrypt'):
    U = len(blocks)
    R = sequence_R(A["A_a2_primary"], X0["X0_a2_primary"], U, U)
    Rt = sequence_R(A["A_a2_secondary"], X0["X0_a2_secondary"], U, U)
    perm = blocks.copy()
    if mode == 'encrypt':
        for i in range(U): perm[i], perm[R[i]] = perm[R[i]].copy(), perm[i].copy()
        for i in range(U): perm[i], perm[Rt[i]] = perm[Rt[i]].copy(), perm[i].copy()
    else:
        for i in reversed(range(U)): perm[i], perm[Rt[i]] = perm[Rt[i]].copy(), perm[i].copy()
        for i in reversed(range(U)): perm[i], perm[R[i]] = perm[R[i]].copy(), perm[i].copy()
    return perm

def mask_process(blocks, omega, psi, mode='encrypt'):
    U = len(blocks)
    l = len(blocks[0])
    out = []
    prev = np.zeros_like(blocks[0], dtype=np.uint8)
    for i in range(U):
        current_block = blocks[i]
        T = ((omega[i % omega.shape[0]].astype(np.uint16)) ^ (psi[i % psi.shape[0]].astype(np.uint16))) % 256
        shift = int(T[0] % l)
        if mode == 'encrypt':
            shifted = np.roll(current_block, shift, axis=1).astype(np.uint16)
            masked = (shifted ^ prev) % 256
            res_block = masked.astype(np.uint8)
            prev = res_block.copy()
        else:
            C = current_block.astype(np.uint16)
            shifted = (C ^ prev.astype(np.uint16)) % 256
            res_block = np.roll(shifted, -shift, axis=1).astype(np.uint8)
            prev = current_block.copy()
        out.append(res_block)
    return np.array(out)

def prepare_image(im, l):
    h, w = im.shape
    h_new = h - (h % l)
    w_new = w - (w % l)
    im = im[:h_new, :w_new]
    blocks = []
    for i in range(0, h_new, l):
        for j in range(0, w_new, l):
            blocks.append(im[i:i+l, j:j+l].copy())
    return np.array(blocks), (h_new, w_new)

def blocks_to_image(blocks, shape, l):
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    idx = 0
    for i in range(0, h, l):
        for j in range(0, w, l):
            out[i:i+l, j:j+l] = blocks[idx]
            idx += 1
    return out

# FUNCIÓN PRINCIPAL CON PASOS
def run_algorithm_with_steps(image, seed, mode='encrypt'):
    rounds = 3
    l = 16
    
    A, X0 = get_deterministic_keys(seed)
    blocks, new_shape = prepare_image(image, l)
    omega = get_omega(A["A_omega"], X0["X0_omega"], l, l)
    psi = get_psi(A["A_psi"], X0["X0_psi"], rounds, l)
    
    # Historial de pasos: [(Nombre_Paso, Imagen_Resultante)]
    history = []
    
    # Paso 0: Estado inicial
    current_img = blocks_to_image(blocks, new_shape, l)
    history.append(("Estado Inicial", current_img))
    
    if mode == 'encrypt':
        for r in range(rounds):
            # Fase 1: Barajado
            blocks = shuffle(blocks, X0, A, 'encrypt')
            img_step = blocks_to_image(blocks, new_shape, l)
            history.append((f"Ronda {r+1}: Barajado (Shuffling)", img_step))
            
            # Fase 2: Enmascaramiento
            blocks = mask_process(blocks, omega, psi, 'encrypt')
            img_step = blocks_to_image(blocks, new_shape, l)
            history.append((f"Ronda {r+1}: Enmascarado (Masking)", img_step))
            
    else: # Decrypt
        for r in range(rounds):
            # Fase 1: Des-enmascaramiento
            blocks = mask_process(blocks, omega, psi, 'decrypt')
            img_step = blocks_to_image(blocks, new_shape, l)
            history.append((f"Ronda {rounds-r}: Des-enmascarado", img_step))
            
            # Fase 2: Des-barajado
            blocks = shuffle(blocks, X0, A, 'decrypt')
            img_step = blocks_to_image(blocks, new_shape, l)
            history.append((f"Ronda {rounds-r}: Des-barajado", img_step))
            
    return history

# ==============================================================================
# 2. INTERFAZ GRÁFICA (Con Slider Mejorado)
# ==============================================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Cifrado Médico - Visualizador de Rondas")
        self.geometry("1100x750")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.setup_tab("Cifrar")
        self.setup_tab("Descifrar")
        
        # Almacenamiento de datos
        self.img_cv_source = {}
        self.history_steps = {}

    def setup_tab(self, name):
        tab = self.tab_view.add(name)
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=3)
        
        # --- Panel Izquierdo (Controles) ---
        frame_controls = ctk.CTkFrame(tab)
        frame_controls.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(frame_controls, text=f"Módulo {name}", font=("Arial", 18, "bold")).pack(pady=15)
        
        ctk.CTkButton(frame_controls, text="Cargar Imagen", command=lambda: self.load_image(name)).pack(pady=10)
        
        ctk.CTkLabel(frame_controls, text="Semilla (Seed):").pack(pady=(20,0))
        entry_seed = ctk.CTkEntry(frame_controls, show="*")
        entry_seed.pack(pady=5)
        
        # Guardar referencia al entry
        setattr(self, f"entry_seed_{name}", entry_seed)
        
        ctk.CTkButton(frame_controls, text=f"Ejecutar {name}", 
                      fg_color="green",
                      command=lambda: self.process(name)).pack(pady=20)
        
        ctk.CTkButton(frame_controls, text="Guardar Imagen Actual", command=lambda: self.save_current_image(name)).pack(pady=10)
        
        # --- Panel Derecho (Visualización) ---
        frame_view = ctk.CTkFrame(tab)
        frame_view.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # 1. SLIDER (Abajo, fijo)
        frame_slider = ctk.CTkFrame(frame_view, fg_color="transparent")
        frame_slider.pack(side="top", fill="x", padx=20, pady=20)
        
        lbl_step_name = ctk.CTkLabel(frame_slider, text="Estado: Esperando...")
        lbl_step_name.pack()
        setattr(self, f"lbl_step_{name}", lbl_step_name)
        
        slider = ctk.CTkSlider(frame_slider, from_=0, to=1, number_of_steps=1, state="disabled")
        slider.pack(fill="x", pady=5)
        # Configurar comando del slider
        slider.configure(command=lambda val, m=name: self.on_slider_change(m, val))
        
        setattr(self, f"slider_{name}", slider)

        # 2. IMAGEN (Arriba, expandible)
        lbl_img = ctk.CTkLabel(frame_view, text="Vista Previa")
        lbl_img.pack(side="top", expand=True, fill="both", pady=10)
        setattr(self, f"lbl_preview_{name}", lbl_img)

    def load_image(self, tab_name):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp")])
        if path:
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            self.img_cv_source[tab_name] = img
            self.show_image(img, tab_name)
            
            # Resetear slider e historial
            self.history_steps[tab_name] = None
            slider = getattr(self, f"slider_{tab_name}")
            slider.configure(state="disabled", from_=0, to=1, number_of_steps=1)
            slider.set(0)
            getattr(self, f"lbl_step_{tab_name}").configure(text="Imagen cargada (Original)")

    def show_image(self, cv_img, tab_name):
        if cv_img is None: return
        img_pil = Image.fromarray(cv_img)
        
        # Ajustar tamaño manteniendo proporción
        target_size = (600, 500)
        img_pil.thumbnail(target_size)
        
        img_tk = ImageTk.PhotoImage(img_pil)
        lbl = getattr(self, f"lbl_preview_{tab_name}")
        lbl.configure(image=img_tk, text="")
        lbl.image = img_tk

    def process(self, mode):
        if mode not in self.img_cv_source:
            messagebox.showerror("Error", "Carga una imagen primero.")
            return
        
        seed = getattr(self, f"entry_seed_{mode}").get()
        if not seed:
            messagebox.showerror("Error", "Introduce una semilla.")
            return
            
        try:
            op = 'encrypt' if mode == "Cifrar" else 'decrypt'
            # Ejecutar algoritmo y obtener historial
            history = run_algorithm_with_steps(self.img_cv_source[mode], seed, mode=op)
            self.history_steps[mode] = history
            
            # Configurar slider
            total_steps = len(history)
            slider = getattr(self, f"slider_{mode}")
            slider.configure(state="normal", from_=0, to=total_steps-1, number_of_steps=total_steps-1)
            slider.set(total_steps-1) # Ir al final por defecto
            
            # Mostrar resultado final
            self.on_slider_change(mode, total_steps-1)
            
            messagebox.showinfo("Éxito", f"Proceso terminado.\nUsa el slider inferior para ver el progreso.")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_slider_change(self, mode, value):
        if not self.history_steps.get(mode): return
        
        idx = int(value)
        # Asegurar que el índice no se salga
        idx = max(0, min(idx, len(self.history_steps[mode]) - 1))
        
        step_name, step_img = self.history_steps[mode][idx]
        
        # Actualizar imagen
        self.show_image(step_img, mode)
        
        # Actualizar etiqueta
        lbl = getattr(self, f"lbl_step_{mode}")
        lbl.configure(text=f"Paso {idx}: {step_name}")

    def save_current_image(self, mode):
        if not self.history_steps.get(mode): return
        
        slider = getattr(self, f"slider_{mode}")
        idx = int(slider.get())
        _, img_to_save = self.history_steps[mode][idx]
        
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            cv.imwrite(path, img_to_save)

if __name__ == "__main__":
    app = App()
    app.mainloop()