import json
import os
import copy
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

# ===================== CARREGAMENTO =====================

def load_odometria(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    odometria     = data.get('odometria', [])
    fps           = data.get('fps', 30.0)
    frame_inicial = 0

    if 'historico_pontos' in data and data['historico_pontos']:
        frames_hist = [int(k) for k in data['historico_pontos'].keys()]
        if frames_hist:
            frame_inicial = min(frames_hist)

    if frame_inicial == 0 and odometria:
        primeiro_tempo_ms = odometria[0][0]
        frame_inicial = int((primeiro_tempo_ms / 1000.0) * fps)

    dados_frame = []
    for tempo_ms, dist_px in odometria:
        frame_relativo = int((tempo_ms / 1000.0) * fps)
        frame_absoluto = frame_inicial + frame_relativo
        dados_frame.append([frame_absoluto, dist_px])

    return dados_frame, data, frame_inicial


# ===================== LÓGICA DE EDIÇÃO =====================

def aplicar_remocoes(dados_originais, regioes):
    dados = copy.deepcopy(dados_originais)
    for (fi, ff) in sorted(regioes, key=lambda r: r[0]):
        dist_inicio = interpolar(dados, fi)
        dist_fim    = interpolar(dados, ff)
        delta       = dist_fim - dist_inicio
        novos = []
        for frame, dist in dados:
            if fi <= frame <= ff:
                novos.append([frame, dist_inicio])
            elif frame > ff:
                novos.append([frame, dist - delta])
            else:
                novos.append([frame, dist])
        dados = novos
    return dados


def interpolar(dados, frame):
    if not dados:
        return 0.0
    if frame <= dados[0][0]:
        return dados[0][1]
    if frame >= dados[-1][0]:
        return dados[-1][1]
    for i in range(len(dados) - 1):
        f1, d1 = dados[i]
        f2, d2 = dados[i + 1]
        if f1 <= frame <= f2:
            t = (frame - f1) / (f2 - f1)
            return d1 + t * (d2 - d1)
    return dados[-1][1]


# ===================== CONSTANTES DE LAYOUT =====================

MARGIN_LEFT   = 65
MARGIN_RIGHT  = 20
MARGIN_TOP    = 40
MARGIN_BOTTOM = 45
GRAPH_H       = 260
VIDEO_H       = 340
VIDEO_W       = 520


# ===================== APLICAÇÃO =====================

class EditorOdometria:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor de Odometria")
        self.root.configure(bg="#0d1117")
        self.root.geometry("1150x820")
        self.root.minsize(900, 700)

        # ── dados ──
        self.json_path       = None
        self.dados_originais = []
        self.dados_editados  = []
        self.json_data       = {}
        self.frame_inicial   = 0
        self.regioes         = []
        self.sel_inicio      = None
        self.sel_fim_temp    = None
        self.canvas_w        = 560

        # ── vídeo ──
        self.cap           = None
        self.video_path    = None
        self.total_frames  = 0
        self.fps_video     = 30.0
        self.current_frame = 0
        self.playing       = False
        self._play_job     = None
        self._photo        = None

        self._build_ui()
        self.root.bind("<KeyPress>", self._on_key)
        self.root.focus_set()

    # ─────────────────────────── UI ───────────────────────────────────────

    def _build_ui(self):
        # ── Barra superior ──
        top = tk.Frame(self.root, bg="#161b22", pady=7, padx=12)
        top.pack(fill="x")

        def tbtn(text, cmd, bg, side="left", px=4):
            b = tk.Button(top, text=text, command=cmd,
                          bg=bg, fg="#e6edf3",
                          activebackground=bg,
                          font=("Consolas", 9, "bold"),
                          relief="flat", padx=10, pady=4,
                          cursor="hand2", borderwidth=0)
            b.pack(side=side, padx=px)
            return b

        tbtn("📂 JSON",    self._abrir_json,   "#0d419d")
        tbtn("🎬 Vídeo",   self._abrir_video,  "#1f4e79")

        self.lbl_arquivo = tk.Label(top, text="Nenhum arquivo",
                                    bg="#161b22", fg="#484f58",
                                    font=("Consolas", 8))
        self.lbl_arquivo.pack(side="left", padx=10)

        tbtn("💾 Salvar",  self._salvar,        "#145a32", side="right")
        tbtn("🔄 Resetar", self._resetar,       "#5d2e00", side="right")
        tbtn("↩ Desfazer", self._desfazer,      "#3b1f6e", side="right")

        # ── Dica de teclas ──
        tk.Label(self.root,
                 text="  ← / → : 1 frame   |   Shift + ← / → : 10 frames"
                      "   |   Espaço : pausar/reproduzir"
                      "   |   Arraste no gráfico para marcar região de remoção",
                 bg="#0d1117", fg="#30363d",
                 font=("Consolas", 8), anchor="w"
                 ).pack(fill="x", pady=(2, 0))

        # ── Corpo: vídeo (esq) | gráfico (dir) ──
        body = tk.Frame(self.root, bg="#0d1117")
        body.pack(fill="both", expand=True, padx=8, pady=6)

        # ---- coluna esquerda: vídeo + controles ----
        left = tk.Frame(body, bg="#0d1117")
        left.pack(side="left", fill="y", padx=(0, 8))

        # moldura do vídeo
        video_frame = tk.Frame(left, bg="#010409",
                               width=VIDEO_W, height=VIDEO_H,
                               relief="flat", borderwidth=0)
        video_frame.pack_propagate(False)
        video_frame.pack()

        self.video_label = tk.Label(video_frame, bg="#010409")
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        self.lbl_sem_video = tk.Label(video_frame,
                                      text="Abra um vídeo\n(🎬 Vídeo)",
                                      bg="#010409", fg="#21262d",
                                      font=("Consolas", 12))
        self.lbl_sem_video.place(relx=0.5, rely=0.5, anchor="center")

        # slider de posição
        self.slider_var = tk.IntVar(value=0)
        self.slider = tk.Scale(left, from_=0, to=1000,
                               orient="horizontal",
                               variable=self.slider_var,
                               command=self._on_slider,
                               bg="#161b22", fg="#484f58",
                               troughcolor="#21262d",
                               highlightthickness=0,
                               sliderrelief="flat",
                               bd=0, length=VIDEO_W,
                               showvalue=False)
        self.slider.pack(pady=(4, 0))

        # botões de controle
        ctrl = tk.Frame(left, bg="#0d1117")
        ctrl.pack(pady=4)

        def cbtn(text, cmd, w=5):
            return tk.Button(ctrl, text=text, command=cmd,
                             bg="#21262d", fg="#c9d1d9",
                             activebackground="#30363d",
                             font=("Consolas", 11),
                             relief="flat", width=w,
                             cursor="hand2", borderwidth=0, pady=3)

        cbtn("⏮", lambda: self._saltar_frames(-10)).pack(side="left", padx=2)
        cbtn("◀",  lambda: self._saltar_frames(-1)).pack(side="left", padx=2)
        self.btn_play = cbtn("▶", self._toggle_play, w=7)
        self.btn_play.pack(side="left", padx=2)
        cbtn("▶",  lambda: self._saltar_frames(1)).pack(side="left", padx=2)
        cbtn("⏭", lambda: self._saltar_frames(10)).pack(side="left", padx=2)

        self.lbl_frame_info = tk.Label(ctrl, text="— / —",
                                       bg="#0d1117", fg="#484f58",
                                       font=("Consolas", 9))
        self.lbl_frame_info.pack(side="left", padx=10)

        # ---- coluna direita: gráfico ----
        right = tk.Frame(body, bg="#0d1117")
        right.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(right, bg="#0d1117",
                                highlightthickness=1,
                                highlightbackground="#21262d")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Configure>",       self._on_resize)
        self.canvas.bind("<Motion>",          self._on_mouse_move)

        # ── Painel inferior ──
        bot = tk.Frame(self.root, bg="#161b22", pady=5, padx=12)
        bot.pack(fill="x")

        tk.Label(bot, text="Regiões:",
                 bg="#161b22", fg="#8b949e",
                 font=("Consolas", 8, "bold")).pack(side="left")
        self.lbl_regioes = tk.Label(bot, text="—",
                                    bg="#161b22", fg="#f85149",
                                    font=("Consolas", 8))
        self.lbl_regioes.pack(side="left", padx=8)

        self.lbl_cursor = tk.Label(bot, text="",
                                   bg="#161b22", fg="#388bfd",
                                   font=("Consolas", 8))
        self.lbl_cursor.pack(side="right")

    # ─────────────────────────── ABRIR ARQUIVOS ───────────────────────────

    def _abrir_json(self):
        path = filedialog.askopenfilename(
            title="Selecione o JSON de odometria",
            filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            dados, json_data, frame_inicial = load_odometria(path)
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar:\n{e}")
            return

        self.json_path       = path
        self.dados_originais = dados
        self.dados_editados  = copy.deepcopy(dados)
        self.json_data       = json_data
        self.frame_inicial   = frame_inicial
        self.regioes         = []
        self.sel_inicio      = None
        self.sel_fim_temp    = None

        self.lbl_arquivo.config(
            text=os.path.basename(path)[-50:], fg="#58a6ff")
        self._atualizar_lista_regioes()

        # se vídeo já está aberto, posiciona no frame inicial
        if self.cap:
            self.current_frame = self.frame_inicial
            self._sincronizar_slider()
            self._mostrar_frame(self.current_frame)
        else:
            self._redesenhar()

    def _abrir_video(self):
        path = filedialog.askopenfilename(
            title="Selecione o vídeo",
            filetypes=[("Vídeo", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir o vídeo.")
            return

        if self.cap:
            self._parar()
            self.cap.release()

        self.cap          = cap
        self.video_path   = path
        self.fps_video    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = self.frame_inicial

        self.lbl_sem_video.place_forget()
        self._sincronizar_slider()
        self._mostrar_frame(self.current_frame)

    # ─────────────────────────── VÍDEO ────────────────────────────────────

    def _sincronizar_slider(self):
        f_min = self.frame_inicial
        f_max = (self.total_frames - 1) if self.cap else 1000
        self.slider.config(from_=f_min, to=f_max)
        self.slider_var.set(self.current_frame)

    def _mostrar_frame(self, frame_idx):
        if not self.cap:
            return
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = self.cap.read()
        if not ret:
            return

        self.current_frame = frame_idx
        self.slider_var.set(frame_idx)
        self.lbl_frame_info.config(
            text=f"{frame_idx} / {self.total_frames - 1}")

        # Redimensionar mantendo proporção dentro da moldura VIDEO_W × VIDEO_H
        h, w = frame_bgr.shape[:2]
        scale = min(VIDEO_W / w, VIDEO_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized    = cv2.resize(frame_bgr, (nw, nh))
        img_rgb    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_pil    = Image.fromarray(img_rgb)
        self._photo = ImageTk.PhotoImage(img_pil)
        self.video_label.config(image=self._photo)

        self._redesenhar()

    def _saltar_frames(self, delta):
        self._parar()
        self._mostrar_frame(self.current_frame + delta)

    def _toggle_play(self):
        if self.playing:
            self._parar()
        else:
            self._iniciar_play()

    def _iniciar_play(self):
        if not self.cap:
            return
        self.playing = True
        self.btn_play.config(text="⏸")
        self._loop_play()

    def _parar(self):
        self.playing = False
        self.btn_play.config(text="▶")
        if self._play_job:
            self.root.after_cancel(self._play_job)
            self._play_job = None

    def _loop_play(self):
        if not self.playing:
            return
        next_f = self.current_frame + 1
        # Limite: fim da odometria ou fim do vídeo
        f_max = self.total_frames - 1
        if self.dados_originais:
            f_max = min(f_max, self.dados_originais[-1][0])
        if next_f > f_max:
            self._parar()
            return
        self._mostrar_frame(next_f)
        delay = max(1, int(1000 / self.fps_video))
        self._play_job = self.root.after(delay, self._loop_play)

    def _on_slider(self, val):
        frame = int(float(val))
        if frame != self.current_frame:
            self._parar()
            self._mostrar_frame(frame)

    # ─────────────────────────── TECLAS ───────────────────────────────────

    def _on_key(self, event):
        key = event.keysym
        if key == "space":
            self._toggle_play()
        elif key == "Right":
            self._saltar_frames(10 if (event.state & 0x0001) else 1)
        elif key == "Left":
            self._saltar_frames(-10 if (event.state & 0x0001) else -1)

    # ─────────────────────────── GRÁFICO — EVENTOS ────────────────────────

    def _on_resize(self, event):
        self.canvas_w = event.width
        self._redesenhar()

    def _on_press(self, event):
        if not self.dados_editados:
            return
        frame = self._x_to_frame(event.x)
        if frame is not None:
            self.sel_inicio   = frame
            self.sel_fim_temp = frame

    def _on_drag(self, event):
        if self.sel_inicio is None:
            return
        frame = self._x_to_frame(event.x)
        if frame is not None:
            self.sel_fim_temp = frame
            self._redesenhar()

    def _on_release(self, event):
        if self.sel_inicio is None:
            return
        frame = self._x_to_frame(event.x)
        if frame is not None:
            fi = min(self.sel_inicio, frame)
            ff = max(self.sel_inicio, frame)
            if ff > fi:
                self.regioes.append((fi, ff))
                self._recalcular()
        self.sel_inicio   = None
        self.sel_fim_temp = None
        self._redesenhar()
        self._atualizar_lista_regioes()

    def _on_mouse_move(self, event):
        frame = self._x_to_frame(event.x)
        if frame is not None and self.dados_editados:
            dist = interpolar(self.dados_editados, frame)
            self.lbl_cursor.config(
                text=f"frame {frame}  |  dist {dist:.1f} px")
        else:
            self.lbl_cursor.config(text="")

    # ─────────────────────────── AÇÕES DE EDIÇÃO ──────────────────────────

    def _recalcular(self):
        self.dados_editados = aplicar_remocoes(
            self.dados_originais, self.regioes)

    def _desfazer(self):
        if self.regioes:
            self.regioes.pop()
            self._recalcular()
            self._atualizar_lista_regioes()
            self._redesenhar()

    def _resetar(self):
        if not self.dados_originais:
            return
        self.regioes        = []
        self.dados_editados = copy.deepcopy(self.dados_originais)
        self._atualizar_lista_regioes()
        self._redesenhar()

    def _salvar(self):
        if not self.json_path or not self.dados_editados:
            messagebox.showwarning("Aviso", "Nenhum dado para salvar.")
            return

        fps = self.json_data.get('fps', 30.0)
        nova_odometria = []
        for frame_abs, dist_px in self.dados_editados:
            frame_relativo = frame_abs - self.frame_inicial
            tempo_ms = int((frame_relativo / fps) * 1000)
            nova_odometria.append([tempo_ms, dist_px])

        novo_json = copy.deepcopy(self.json_data)
        novo_json['odometria'] = nova_odometria

        base = self.json_path
        if base.endswith(".json"):
            base = base[:-5]
        output_path = base + "_revisado.json"

        with open(output_path, 'w') as f:
            json.dump(novo_json, f, indent=2)

        messagebox.showinfo("Salvo!", f"Arquivo salvo em:\n{output_path}")

    def _atualizar_lista_regioes(self):
        if not self.regioes:
            self.lbl_regioes.config(text="—")
        else:
            partes = [f"[{fi}→{ff}]" for fi, ff in self.regioes]
            self.lbl_regioes.config(text="  ".join(partes))

    # ─────────────────────────── GRÁFICO — DESENHO ────────────────────────

    def _graph_bounds(self):
        w  = max(self.canvas_w, 200)
        gx = MARGIN_LEFT
        gy = MARGIN_TOP
        gw = w - MARGIN_LEFT - MARGIN_RIGHT
        gh = GRAPH_H - MARGIN_TOP - MARGIN_BOTTOM
        return gx, gy, gw, gh

    def _frame_to_x(self, frame):
        gx, gy, gw, gh = self._graph_bounds()
        dados = self.dados_originais
        if not dados:
            return gx
        f_min = self.frame_inicial
        f_max = dados[-1][0]
        if f_max == f_min:
            return gx
        return gx + int(((frame - f_min) / (f_max - f_min)) * gw)

    def _x_to_frame(self, x):
        gx, gy, gw, gh = self._graph_bounds()
        dados = self.dados_originais
        if not dados:
            return None
        f_min = self.frame_inicial
        f_max = dados[-1][0]
        t = (x - gx) / gw
        t = max(0.0, min(1.0, t))
        return int(f_min + t * (f_max - f_min))

    def _dist_to_y(self, dist, max_dist):
        gx, gy, gw, gh = self._graph_bounds()
        if max_dist == 0:
            return gy + gh
        return gy + int((1 - dist / max_dist) * gh)

    def _redesenhar(self):
        c = self.canvas
        c.delete("all")

        gx, gy, gw, gh = self._graph_bounds()
        dados_orig = self.dados_originais
        dados_edit = self.dados_editados

        if not dados_orig:
            c.create_text(gx + gw // 2, gy + gh // 2,
                          text="Abra um JSON de odometria",
                          fill="#30363d", font=("Consolas", 12))
            return

        max_dist    = max(d[1] for d in dados_orig) * 1.1 or 1.0
        f_min       = self.frame_inicial
        f_max       = dados_orig[-1][0]
        frame_range = f_max - f_min or 1

        # ── Fundo ──
        c.create_rectangle(gx, gy, gx + gw, gy + gh,
                           fill="#0d1117", outline="#21262d")

        # ── Grade horizontal ──
        for i in range(7):
            y   = gy + int(i * gh / 6)
            val = max_dist * (1 - i / 6)
            c.create_line(gx, y, gx + gw, y, fill="#161b22", dash=(3, 4))
            c.create_text(gx - 6, y, text=f"{int(val)}",
                          anchor="e", fill="#3d444d", font=("Consolas", 7))

        # ── Grade vertical ──
        for i in range(11):
            x  = gx + int(i * gw / 10)
            fv = f_min + int(i * frame_range / 10)
            c.create_line(x, gy, x, gy + gh, fill="#161b22", dash=(3, 4))
            c.create_text(x, gy + gh + 13, text=str(fv),
                          fill="#3d444d", font=("Consolas", 7))

        # ── Rótulos dos eixos ──
        c.create_text(gx - 52, gy + gh // 2,
                      text="dist (px)", fill="#3d444d",
                      font=("Consolas", 7), angle=90)
        c.create_text(gx + gw // 2, gy + gh + 30,
                      text="frame", fill="#3d444d", font=("Consolas", 7))

        # ── Regiões confirmadas (vermelho) ──
        for (fi, ff) in self.regioes:
            x1 = self._frame_to_x(fi)
            x2 = self._frame_to_x(ff)
            c.create_rectangle(x1, gy, x2, gy + gh,
                               fill="#3d0000", outline="#da3633",
                               stipple="gray25")
            c.create_line(x1, gy, x1, gy + gh, fill="#da3633", width=1)
            c.create_line(x2, gy, x2, gy + gh, fill="#da3633", width=1)

        # ── Seleção em andamento (azul) ──
        if self.sel_inicio is not None and self.sel_fim_temp is not None:
            xa = self._frame_to_x(min(self.sel_inicio, self.sel_fim_temp))
            xb = self._frame_to_x(max(self.sel_inicio, self.sel_fim_temp))
            c.create_rectangle(xa, gy, xb, gy + gh,
                               fill="#001a33", outline="#388bfd",
                               stipple="gray25")
            c.create_line(xa, gy, xa, gy + gh,
                          fill="#388bfd", width=1, dash=(4, 3))
            c.create_line(xb, gy, xb, gy + gh,
                          fill="#388bfd", width=1, dash=(4, 3))

        # ── Curva original (azul apagado) ──
        pts_orig = []
        for frame, dist in dados_orig:
            pts_orig.append(self._frame_to_x(frame))
            pts_orig.append(self._dist_to_y(dist, max_dist))
        if len(pts_orig) >= 4:
            c.create_line(*pts_orig, fill="#1f4e79", width=1)

        # ── Curva editada (verde) ──
        pts_edit = []
        for frame, dist in dados_edit:
            pts_edit.append(self._frame_to_x(frame))
            pts_edit.append(self._dist_to_y(dist, max_dist))
        if len(pts_edit) >= 4:
            c.create_line(*pts_edit, fill="#3fb950", width=2)

        # ── Barra vertical amarela do frame atual do vídeo ──
        if self.cap is not None:
            vx = self._frame_to_x(self.current_frame)
            # sombra suave
            c.create_line(vx - 1, gy, vx - 1, gy + gh,
                          fill="#40360a", width=3)
            c.create_line(vx + 1, gy, vx + 1, gy + gh,
                          fill="#40360a", width=3)
            # linha principal
            c.create_line(vx, gy, vx, gy + gh,
                          fill="#e3b341", width=2)
            # triângulo no topo
            c.create_polygon(vx - 6, gy - 2,
                             vx + 6, gy - 2,
                             vx,     gy + 9,
                             fill="#e3b341", outline="")
            # rótulo
            label_x = vx + 8 if vx < gx + gw - 60 else vx - 8
            anchor   = "w"    if vx < gx + gw - 60 else "e"
            c.create_text(label_x, gy - 14,
                          text=str(self.current_frame),
                          fill="#e3b341", font=("Consolas", 7, "bold"),
                          anchor=anchor)

        # ── Título ──
        c.create_text(gx, gy - 24,
                      text="ODOMETRIA  —  arraste para marcar região de remoção",
                      anchor="w", fill="#388bfd",
                      font=("Consolas", 8, "bold"))

        # ── Legenda ──
        lx = gx + gw - 10
        c.create_line(lx - 28, gy + 12, lx, gy + 12, fill="#1f4e79", width=2)
        c.create_text(lx - 32, gy + 12, text="original",
                      anchor="e", fill="#1f6eb5", font=("Consolas", 7))
        c.create_line(lx - 28, gy + 24, lx, gy + 24, fill="#3fb950", width=2)
        c.create_text(lx - 32, gy + 24, text="editado",
                      anchor="e", fill="#3fb950", font=("Consolas", 7))
        if self.cap:
            c.create_line(lx - 28, gy + 36, lx, gy + 36,
                          fill="#e3b341", width=2)
            c.create_text(lx - 32, gy + 36, text="frame atual",
                          anchor="e", fill="#e3b341", font=("Consolas", 7))


# ===================== MAIN =====================

if __name__ == "__main__":
    root = tk.Tk()
    app  = EditorOdometria(root)
    root.mainloop()