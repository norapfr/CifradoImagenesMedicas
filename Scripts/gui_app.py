import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv
import json
import os

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

def generate_random_keys():
    params = {
        "p_omega": np.random.randint(1, 10, 4).tolist(),
        "p_psi": np.random.randint(1, 10, 4).tolist(),
        "p_a2_primary": np.random.randint(1, 10, 2).tolist(),
        "p_a2_secondary": np.random.randint(1, 10, 2).tolist()
    }
    
    def rand_nonzero(n):
        x = np.random.random(n)
        while np.allclose(x, 0): x = np.random.random(n)
        return x.tolist()

    X0 = {
        "X0_omega": rand_nonzero(4),
        "X0_psi": rand_nonzero(4),
        "X0_a2_primary": rand_nonzero(2),
        "X0_a2_secondary": rand_nonzero(2)
    }
    return params, X0

def build_matrices(params):
    return {
        "A_omega": cat_map_4d_matrix(*params["p_omega"]),
        "A_psi": cat_map_4d_matrix(*params["p_psi"]),
        "A_a2_primary": cat_map_2d_matrix(*params["p_a2_primary"]),
        "A_a2_secondary": cat_map_2d_matrix(*params["p_a2_secondary"])
    }

def prng_4d(A, X0, L):
    X = np.array(X0)
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

def run_algorithm_with_steps(image, key_data, mode='encrypt'):
    rounds = 3
    l = 16
    params = key_data["params"]
    X0_raw = key_data["X0"]
    X0 = {k: np.array(v) for k, v in X0_raw.items()}
    A = build_matrices(params)
    
    blocks, new_shape = prepare_image(image, l)
    omega = get_omega(A["A_omega"], X0["X0_omega"], l, l)
    psi = get_psi(A["A_psi"], X0["X0_psi"], rounds, l)
    
    history = []
    current_img = blocks_to_image(blocks, new_shape, l)
    history.append(("Estado Inicial", current_img))
    
    if mode == 'encrypt':
        for r in range(rounds):
            blocks = shuffle(blocks, X0, A, 'encrypt')
            history.append((f"Ronda {r+1}: Barajado", blocks_to_image(blocks, new_shape, l)))
            blocks = mask_process(blocks, omega, psi, 'encrypt')
            history.append((f"Ronda {r+1}: Enmascarado", blocks_to_image(blocks, new_shape, l)))
    else:
        for r in range(rounds):
            blocks = mask_process(blocks, omega, psi, 'decrypt')
            history.append((f"Ronda {rounds-r}: Des-enmascarado", blocks_to_image(blocks, new_shape, l)))
            blocks = shuffle(blocks, X0, A, 'decrypt')
            history.append((f"Ronda {rounds-r}: Des-barajado", blocks_to_image(blocks, new_shape, l)))
            
    return history

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Cifrado Médico - JSON Key System")
        self.geometry("1100x800")
        self.current_keys = None
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.img_cv_source = {}
        self.history_steps = {}

        self.setup_encryption_tab()
        self.setup_decryption_tab()

    def setup_encryption_tab(self):
        tab = self.tab_view.add("Cifrar")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=3)
        frame_controls = ctk.CTkFrame(tab)
        frame_controls.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(frame_controls, text="Módulo de Cifrado", font=("Arial", 18, "bold")).pack(pady=15)
        
        ctk.CTkButton(frame_controls, text="1. Cargar Imagen Original", command=lambda: self.load_image("Cifrar")).pack(pady=10)
        
        ctk.CTkLabel(frame_controls, text="Nota: Se generarán claves nuevas\naleatorias automáticamente.", text_color="gray").pack(pady=10)
        
        ctk.CTkButton(frame_controls, text="2. EJECUTAR CIFRADO", fg_color="green", command=self.run_encryption).pack(pady=10)
        
        self.btn_save_img_enc = ctk.CTkButton(frame_controls, text="3. Guardar Imagen Cifrada", state="disabled", command=lambda: self.save_current_image("Cifrar"))
        self.btn_save_img_enc.pack(pady=10)
        self.btn_save_json = ctk.CTkButton(frame_controls, text="4. Guardar Claves (JSON)", fg_color="#D35400", state="disabled", command=self.save_keys_json)
        self.btn_save_json.pack(pady=10)
        self.setup_visualization_frame(tab, "Cifrar")

    def setup_decryption_tab(self):
        tab = self.tab_view.add("Descifrar")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=3)
        frame_controls = ctk.CTkFrame(tab)
        frame_controls.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(frame_controls, text="Módulo de Descifrado", font=("Arial", 18, "bold")).pack(pady=15)
        
        ctk.CTkButton(frame_controls, text="1. Cargar Imagen Cifrada", command=lambda: self.load_image("Descifrar")).pack(pady=10)
        
        ctk.CTkLabel(frame_controls, text="Gestión de Claves:").pack(pady=(20,5))
        
        self.lbl_key_status = ctk.CTkLabel(frame_controls, text="Estado: Usando últimas claves generadas", text_color="yellow", font=("Arial", 11))
        self.lbl_key_status.pack(pady=5)
        
        ctk.CTkButton(frame_controls, text="Cargar JSON de Claves", fg_color="#2980B9", command=self.load_keys_json).pack(pady=5)
        
        ctk.CTkButton(frame_controls, text="2. EJECUTAR DESCIFRADO", fg_color="green", command=self.run_decryption).pack(pady=20)
        
        self.btn_save_img_dec = ctk.CTkButton(frame_controls, text="3. Guardar Imagen Descifrada", state="disabled", command=lambda: self.save_current_image("Descifrar"))
        self.btn_save_img_dec.pack(pady=10)
        self.setup_visualization_frame(tab, "Descifrar")

    def setup_visualization_frame(self, tab, name):
        frame_view = ctk.CTkFrame(tab)
        frame_view.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        frame_slider = ctk.CTkFrame(frame_view, fg_color="transparent")
        frame_slider.pack(side="top", fill="x", padx=20, pady=20)
        
        lbl_step = ctk.CTkLabel(frame_slider, text="Estado: Esperando...")
        lbl_step.pack()
        setattr(self, f"lbl_step_{name}", lbl_step)
        
        slider = ctk.CTkSlider(frame_slider, from_=0, to=1, number_of_steps=1, state="disabled")
        slider.pack(fill="x", pady=5)
        slider.configure(command=lambda val, m=name: self.on_slider_change(m, val))
        setattr(self, f"slider_{name}", slider)

        lbl_img = ctk.CTkLabel(frame_view, text="Vista Previa")
        lbl_img.pack(side="top", expand=True, fill="both", pady=10)
        setattr(self, f"lbl_preview_{name}", lbl_img)

    def load_image(self, tab_name):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.bmp")])
        if path:
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            self.img_cv_source[tab_name] = img
            self.show_image(img, tab_name)
            self.history_steps[tab_name] = None
            getattr(self, f"slider_{tab_name}").configure(state="disabled")
            getattr(self, f"lbl_step_{tab_name}").configure(text="Imagen Cargada")
            
            if tab_name == "Cifrar":
                self.btn_save_img_enc.configure(state="disabled")
                self.btn_save_json.configure(state="disabled")
            else:
                self.btn_save_img_dec.configure(state="disabled")

    def show_image(self, cv_img, tab_name):
        if cv_img is None: return
        img_pil = Image.fromarray(cv_img)
        img_pil.thumbnail((600, 500))
        img_tk = ImageTk.PhotoImage(img_pil)
        lbl = getattr(self, f"lbl_preview_{tab_name}")
        lbl.configure(image=img_tk, text="")
        lbl.image = img_tk

    def run_encryption(self):
        if "Cifrar" not in self.img_cv_source:
            messagebox.showerror("Error", "Carga una imagen primero.")
            return

        try:
            params, X0 = generate_random_keys()
            self.current_keys = {"params": params, "X0": X0}
            self.lbl_key_status.configure(text="Estado: Usando claves recién generadas (Memoria)", text_color="#2ECC71")
            history = run_algorithm_with_steps(self.img_cv_source["Cifrar"], self.current_keys, mode='encrypt')
            self.history_steps["Cifrar"] = history
            self.configure_slider("Cifrar", len(history))
            self.btn_save_img_enc.configure(state="normal")
            self.btn_save_json.configure(state="normal")
            
            messagebox.showinfo("Éxito", "Imagen cifrada. \n¡No olvides guardar el archivo JSON de claves!")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_keys_json(self):
        if not self.current_keys: return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="keys.json")
        if path:
            with open(path, 'w') as f:
                json.dump(self.current_keys, f, indent=4)
            messagebox.showinfo("Guardado", "Claves exportadas correctamente.")

    def load_keys_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                if "params" in data and "X0" in data:
                    self.current_keys = data
                    self.lbl_key_status.configure(text=f"Estado: Claves cargadas desde {os.path.basename(path)}", text_color="#3498DB")
                    messagebox.showinfo("Cargado", "Claves importadas correctamente.")
                else:
                    messagebox.showerror("Error", "El archivo JSON no tiene el formato correcto.")
            except Exception as e:
                messagebox.showerror("Error", f"Fallo al leer JSON: {str(e)}")

    def run_decryption(self):
        if "Descifrar" not in self.img_cv_source:
            messagebox.showerror("Error", "Carga una imagen cifrada primero.")
            return
        
        if self.current_keys is None:
            messagebox.showerror("Error Falta Clave", "No hay claves cargadas.\nPor favor carga un archivo JSON o cifra una imagen antes.")
            return

        try:
            history = run_algorithm_with_steps(self.img_cv_source["Descifrar"], self.current_keys, mode='decrypt')
            self.history_steps["Descifrar"] = history
            
            self.configure_slider("Descifrar", len(history))
            self.btn_save_img_dec.configure(state="normal")
            
            messagebox.showinfo("Éxito", "Imagen descifrada.")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def configure_slider(self, tab_name, total_steps):
        slider = getattr(self, f"slider_{tab_name}")
        slider.configure(state="normal", from_=0, to=total_steps-1, number_of_steps=total_steps-1)
        slider.set(total_steps-1)
        self.on_slider_change(tab_name, total_steps-1)

    def on_slider_change(self, mode, value):
        if not self.history_steps.get(mode): return
        idx = int(value)
        idx = max(0, min(idx, len(self.history_steps[mode]) - 1))
        
        step_name, step_img = self.history_steps[mode][idx]
        self.show_image(step_img, mode)
        getattr(self, f"lbl_step_{mode}").configure(text=f"Paso {idx}: {step_name}")

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