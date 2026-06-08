import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def estimar_K(largura: int, altura: int, fov_horizontal_graus: float = None) -> np.ndarray:
    """
    Estima a matriz intrínseca K.
    Se fov_horizontal_graus não for informado, usa heurística f ≈ max(w, h).
    """
    cx = largura / 2.0
    cy = altura / 2.0

    if fov_horizontal_graus:
        fov_rad = np.deg2rad(fov_horizontal_graus)
        fx = (largura / 2.0) / np.tan(fov_rad / 2.0)
    else:
        fx = max(largura, altura)  # heurística comum

    fy = fx  # assumindo pixels quadrados

    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    return K


def salvar_K(K: np.ndarray, largura: int, altura: int,
             numero_frame: int, caminho_saida: str):
    dados = {
        "resolucao": {"largura": largura, "altura": altura},
        "numero_frame": numero_frame,
        "matriz_K": K.tolist(),
        "fx": K[0, 0], "fy": K[1, 1],
        "cx": K[0, 2], "cy": K[1, 2],
        "distorcao": [0.0, 0.0, 0.0, 0.0, 0.0]  # k1,k2,p1,p2,k3
    }
    with open(caminho_saida, "w") as f:
        json.dump(dados, f, indent=4)
    print(f"[OK] Matriz K salva em: {caminho_saida}")
    return dados


# ─────────────────────────────────────────────────────────────────────────────
# Interface Gráfica
# ─────────────────────────────────────────────────────────────────────────────

class AppGerarK:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gerador da Matriz K")
        self.root.resizable(False, False)

        self.caminho_video = tk.StringVar()
        self.numero_frame  = tk.IntVar(value=0)
        self.fov_graus     = tk.StringVar(value="")
        self.saida_json    = tk.StringVar(value="matriz_K.json")

        self._construir_ui()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _construir_ui(self):
        pad = dict(padx=10, pady=5)

        # Arquivo de vídeo
        frame_video = ttk.LabelFrame(self.root, text="Vídeo")
        frame_video.grid(row=0, column=0, columnspan=3, sticky="ew", **pad)

        ttk.Entry(frame_video, textvariable=self.caminho_video, width=55).grid(
            row=0, column=0, padx=5, pady=5)
        ttk.Button(frame_video, text="Selecionar…", command=self._selecionar_video).grid(
            row=0, column=1, padx=5, pady=5)

        # Frame número
        ttk.Label(self.root, text="Número do frame:").grid(row=1, column=0, sticky="e", **pad)
        ttk.Entry(self.root, textvariable=self.numero_frame, width=10).grid(
            row=1, column=1, sticky="w", **pad)
        ttk.Button(self.root, text="Pré-visualizar", command=self._previsualizar).grid(
            row=1, column=2, **pad)

        # FOV opcional
        ttk.Label(self.root, text="FOV horizontal (°) [opcional]:").grid(
            row=2, column=0, sticky="e", **pad)
        ttk.Entry(self.root, textvariable=self.fov_graus, width=10).grid(
            row=2, column=1, sticky="w", **pad)
        ttk.Label(self.root, text="← deixe vazio para heurística").grid(
            row=2, column=2, sticky="w", **pad)

        # Arquivo de saída
        ttk.Label(self.root, text="Arquivo de saída (.json):").grid(
            row=3, column=0, sticky="e", **pad)
        ttk.Entry(self.root, textvariable=self.saida_json, width=30).grid(
            row=3, column=1, sticky="w", **pad)
        ttk.Button(self.root, text="…", command=self._escolher_saida, width=3).grid(
            row=3, column=2, sticky="w", **pad)

        # Resultado
        self.txt_resultado = tk.Text(self.root, height=12, width=60,
                                     state="disabled", bg="#f0f0f0", font=("Courier", 9))
        self.txt_resultado.grid(row=4, column=0, columnspan=3, **pad)

        # Botão principal
        ttk.Button(self.root, text="⚙  Gerar Matriz K",
                   command=self._gerar_K).grid(row=5, column=0, columnspan=3, pady=10)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _selecionar_video(self):
        caminho = filedialog.askopenfilename(
            title="Selecionar vídeo",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Todos", "*.*")]
        )
        if caminho:
            self.caminho_video.set(caminho)

    def _escolher_saida(self):
        caminho = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")]
        )
        if caminho:
            self.saida_json.set(caminho)

    def _previsualizar(self):
        if not self.caminho_video.get():
            messagebox.showwarning("Aviso", "Selecione um vídeo primeiro.")
            return
        cap = cv2.VideoCapture(self.caminho_video.get())
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.numero_frame.get())
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Erro", f"Não foi possível ler o frame {self.numero_frame.get()} "
                                          f"(total: {total} frames).")
            return
        cv2.imshow(f"Frame {self.numero_frame.get()} — pressione qualquer tecla para fechar", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _gerar_K(self):
        caminho = self.caminho_video.get()
        if not caminho or not os.path.exists(caminho):
            messagebox.showerror("Erro", "Arquivo de vídeo não encontrado.")
            return

        cap = cv2.VideoCapture(caminho)
        largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        fov_str = self.fov_graus.get().strip()
        fov = float(fov_str) if fov_str else None

        K = estimar_K(largura, altura, fov)
        dados = salvar_K(K, largura, altura, self.numero_frame.get(), self.saida_json.get())

        # Exibir resultado
        texto = (
            f"Resolução : {largura} × {altura}\n"
            f"Frame     : {self.numero_frame.get()}\n"
            f"FOV usado : {'heurística (f=max(w,h))' if fov is None else f'{fov}°'}\n\n"
            f"Matriz K:\n"
            f"  [{K[0,0]:.2f}   0.00   {K[0,2]:.2f}]\n"
            f"  [  0.00  {K[1,1]:.2f}  {K[1,2]:.2f}]\n"
            f"  [  0.00    0.00    1.00]\n\n"
            f"Salvo em: {self.saida_json.get()}"
        )
        self.txt_resultado.config(state="normal")
        self.txt_resultado.delete("1.0", "end")
        self.txt_resultado.insert("end", texto)
        self.txt_resultado.config(state="disabled")
        messagebox.showinfo("Sucesso", f"Matriz K gerada e salva em:\n{self.saida_json.get()}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = AppGerarK(root)
    root.mainloop()