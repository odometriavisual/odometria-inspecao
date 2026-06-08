

import pygame
import cv2
import os
import sys
import tkinter as tk
from tkinter import filedialog
import numpy as np
import threading

# ──────────────────────────────────────────────
#  Constantes
# ──────────────────────────────────────────────
WINDOW_W, WINDOW_H = 1280, 800
BG_COLOR       = (15, 15, 20)
GRID_COLOR     = (40, 40, 50)
PANEL_BG       = (22, 22, 30)
SELECTED_COLOR = (0, 200, 140)
TEXT_COLOR     = (220, 220, 230)
DIM_TEXT       = (100, 100, 120)
CROSSHAIR_COLOR = (255, 220, 80, 80)   # RGBA – será usado via Surface
CROSSHAIR_W    = 1

STEP_ARROW  = 1
STEP_SHIFT  = 10
STEP_CTRL   = 100

FPS_UI      = 30   # taxa de redesenho da interface


# ──────────────────────────────────────────────
#  Classe de vídeo (carregamento assíncrono)
# ──────────────────────────────────────────────
class VideoPlayer:
    def __init__(self, path: str):
        self.path      = path
        self.name      = os.path.basename(path)
        self.cap       = cv2.VideoCapture(path)
        self.total     = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps       = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self._frame    = 0
        self._surface  = None
        self._lock     = threading.Lock()
        self._loaded   = False
        self._load_frame(0)

    # ── internal ──────────────────────────────
    def _read_at(self, idx: int):
        idx = max(0, min(idx, self.total - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _load_frame(self, idx: int):
        rgb = self._read_at(idx)
        if rgb is None:
            return
        with self._lock:
            self._frame   = idx
            self._raw_rgb = rgb
            self._loaded  = True

    # ── public ───────────────────────────────
    def seek(self, delta: int):
        self._load_frame(self._frame + delta)

    def seek_to(self, idx: int):
        self._load_frame(idx)

    @property
    def frame_index(self) -> int:
        return self._frame

    @property
    def time_seconds(self) -> float:
        return self._frame / self.fps if self.fps else 0.0

    def get_surface(self, target_w: int, target_h: int) -> pygame.Surface:
        """Retorna surface redimensionada para target_w x target_h."""
        if not self._loaded:
            s = pygame.Surface((target_w, target_h))
            s.fill((30, 30, 40))
            return s
        with self._lock:
            rgb = self._raw_rgb
        # mantém aspect ratio
        h, w = rgb.shape[:2]
        ratio = min(target_w / w, target_h / h)
        nw, nh = int(w * ratio), int(h * ratio)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        # centraliza em fundo preto
        surf = pygame.Surface((target_w, target_h))
        surf.fill((20, 20, 28))
        px = (target_w - nw) // 2
        py = (target_h - nh) // 2
        frame_surf = pygame.surfarray.make_surface(resized.swapaxes(0, 1))
        surf.blit(frame_surf, (px, py))
        return surf

    def release(self):
        self.cap.release()


# ──────────────────────────────────────────────
#  App principal
# ──────────────────────────────────────────────
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Odometry Sync")
        self.screen = pygame.display.set_mode(
            (WINDOW_W, WINDOW_H), pygame.RESIZABLE
        )
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("monospace", 12)
        self.font_md = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_lg = pygame.font.SysFont("monospace", 18, bold=True)

        self.videos: list[VideoPlayer] = []
        self.selected  = 0          # índice 0-3 do vídeo ativo
        self.fullscreen = False     # True = modo fullscreen
        self.fs_index   = 0         # qual vídeo está em fullscreen
        self.mouse_pos  = (0, 0)

        # Surface semitransparente para crosshair
        self._crosshair_h = None
        self._crosshair_v = None

    # ── seleção de pasta / arquivos ───────────
    def pick_videos(self) -> list[str]:
        root = tk.Tk()
        root.withdraw()
        paths = []
        while len(paths) < 4:
            remaining = 4 - len(paths)
            p = filedialog.askopenfilename(
                title=f"Selecione o vídeo {len(paths)+1} de 4  ({remaining} restante(s))",
                filetypes=[("Vídeos", "*.mp4 *.avi *.mkv *.mov *.wmv *.webm"), ("Todos", "*.*")]
            )
            if not p:
                # usuário cancelou
                if paths:
                    cont = tk.messagebox.askyesno(
                        "Continuar?",
                        f"Só {len(paths)} vídeo(s) selecionado(s). Continuar assim?"
                    )
                    if cont:
                        break
                else:
                    sys.exit(0)
            else:
                paths.append(p)
        root.destroy()
        return paths

    # ── layout de grade ───────────────────────
    def _cell_rect(self, idx: int) -> pygame.Rect:
        """Retorna o rect da célula idx (0-3) para 2x2."""
        W, H = self.screen.get_size()
        margin = 6
        header = 50
        cell_w = (W - margin * 3) // 2
        cell_h = (H - header - margin * 3) // 2
        col = idx % 2
        row = idx // 2
        x = margin + col * (cell_w + margin)
        y = header + margin + row * (cell_h + margin)
        return pygame.Rect(x, y, cell_w, cell_h)

    # ── desenho ───────────────────────────────
    def _draw_grid(self):
        W, H = self.screen.get_size()
        self.screen.fill(BG_COLOR)

        # header
        header_rect = pygame.Rect(0, 0, W, 50)
        pygame.draw.rect(self.screen, PANEL_BG, header_rect)
        title = self.font_lg.render("ODOMETRY SYNC", True, SELECTED_COLOR)
        self.screen.blit(title, (14, 14))
        hint = self.font_sm.render(
            "←/→ move frame | Shift ±10 | Ctrl ±100 | 2× clique = fullscreen | ESPAÇO = print",
            True, DIM_TEXT
        )
        self.screen.blit(hint, (200, 18))

        for i, vid in enumerate(self.videos):
            rect = self._cell_rect(i)
            is_sel = (i == self.selected)

            # borda
            border_col = SELECTED_COLOR if is_sel else GRID_COLOR
            pygame.draw.rect(self.screen, border_col, rect.inflate(4, 4), border_radius=4)

            # frame do vídeo
            vsurf = vid.get_surface(rect.width, rect.height)
            self.screen.blit(vsurf, rect.topleft)

            # overlay de info
            info_bg = pygame.Surface((rect.width, 28), pygame.SRCALPHA)
            info_bg.fill((10, 10, 15, 180))
            self.screen.blit(info_bg, (rect.x, rect.y + rect.height - 28))

            label = self.font_sm.render(
                f"{vid.name}   f:{vid.frame_index}/{vid.total-1}   "
                f"{vid.time_seconds:.3f}s",
                True, SELECTED_COLOR if is_sel else TEXT_COLOR
            )
            self.screen.blit(label, (rect.x + 6, rect.y + rect.height - 22))

            # ícone de seleção
            if is_sel:
                sel_txt = self.font_sm.render("● ATIVO", True, SELECTED_COLOR)
                self.screen.blit(sel_txt, (rect.x + 6, rect.y + 6))

    def _draw_fullscreen(self):
        W, H = self.screen.get_size()
        self.screen.fill(BG_COLOR)

        vid = self.videos[self.fs_index]
        vsurf = vid.get_surface(W, H - 46)
        self.screen.blit(vsurf, (0, 0))

        # barra inferior
        bar = pygame.Surface((W, 46), pygame.SRCALPHA)
        bar.fill((10, 10, 15, 200))
        self.screen.blit(bar, (0, H - 46))

        info = self.font_md.render(
            f"{vid.name}   frame: {vid.frame_index} / {vid.total-1}"
            f"   tempo: {vid.time_seconds:.4f}s   [ESC = voltar]",
            True, SELECTED_COLOR
        )
        self.screen.blit(info, (14, H - 32))

    def _draw_crosshair(self):
        W, H = self.screen.get_size()
        mx, my = self.mouse_pos

        # linha horizontal
        h_surf = pygame.Surface((W, 1), pygame.SRCALPHA)
        h_surf.fill((255, 220, 80, 70))
        self.screen.blit(h_surf, (0, my))

        # linha vertical
        v_surf = pygame.Surface((1, H), pygame.SRCALPHA)
        v_surf.fill((255, 220, 80, 70))
        self.screen.blit(v_surf, (mx, 0))

    # ── eventos ──────────────────────────────
    def _vid_at_pos(self, pos) -> int | None:
        """Retorna índice do vídeo sob o cursor, ou None."""
        for i in range(len(self.videos)):
            if self._cell_rect(i).collidepoint(pos):
                return i
        return None

    def _handle_key(self, event):
        mods = pygame.key.get_mods()
        ctrl  = bool(mods & pygame.KMOD_CTRL)
        shift = bool(mods & pygame.KMOD_SHIFT)

        vid_idx = self.fs_index if self.fullscreen else self.selected
        if vid_idx >= len(self.videos):
            return

        vid = self.videos[vid_idx]

        step = STEP_CTRL if ctrl else (STEP_SHIFT if shift else STEP_ARROW)

        if event.key == pygame.K_RIGHT:
            vid.seek(+step)
        elif event.key == pygame.K_LEFT:
            vid.seek(-step)
        elif event.key == pygame.K_ESCAPE:
            if self.fullscreen:
                self.fullscreen = False
        elif event.key == pygame.K_SPACE:
            self._print_status()

    def _print_status(self):
        import json
        data = {
            os.path.splitext(vid.name)[0]: {
                "frame":   vid.frame_index,
                "tempo_s": round(vid.time_seconds, 4),
                "direcao": "",
            }
            for vid in self.videos
        }
        print("data =", json.dumps(data, indent=4, ensure_ascii=False))

    # ── loop principal ────────────────────────
    def run(self):
        paths = self.pick_videos()
        if not paths:
            return

        print("Carregando vídeos…")
        for p in paths:
            self.videos.append(VideoPlayer(p))
        print("Pronto.")

        _click_time   = [0.0] * 4
        _click_idx    = [-1]

        running = True
        while running:
            dt = self.clock.tick(FPS_UI) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEMOTION:
                    self.mouse_pos = event.pos

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not self.fullscreen:
                        idx = self._vid_at_pos(event.pos)
                        if idx is not None:
                            now = pygame.time.get_ticks() / 1000.0
                            if (
                                idx == _click_idx[0]
                                and now - _click_time[idx] < 0.4
                            ):
                                # duplo clique → fullscreen
                                self.fullscreen = True
                                self.fs_index   = idx
                                _click_time[idx] = 0
                            else:
                                # simples → selecionar
                                self.selected   = idx
                                _click_idx[0]   = idx
                                _click_time[idx] = now

                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event)

                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(
                        event.size, pygame.RESIZABLE
                    )

            # ── desenho ──────────────────────
            if self.fullscreen:
                self._draw_fullscreen()
            else:
                self._draw_grid()

            self._draw_crosshair()
            pygame.display.flip()

        for v in self.videos:
            v.release()
        pygame.quit()


# ──────────────────────────────────────────────
if __name__ == "__main__":
    App().run()