import cv2
import numpy as np
import open3d as o3d
import json
import os
import sys
import tkinter as tk
from tkinter import filedialog
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


# CONFIGURAÇÕES

METODO_K        = "fisico"   # "fisico" | "fov" | "heuristica" | "json"
FOCAL_MM        = 4.25
SENSOR_W_MM     = 6.17
SENSOR_H_MM     = 4.55
CX_OFFSET       = 0.0
CY_OFFSET       = 0.0
FOV_H_GRAUS     = 90.0
CAMINHO_K_JSON  = ""
DISTORCAO       = [0.0, 0.0, 0.0, 0.0, 0.0]

SUBDIVIDIR_MALHA     = True
SUBDIVISAO_ITERACOES = 2
SUBDIVISAO_METODO    = "midpoint"   # "midpoint" | "loop"

IMAN_RAIO_PX     = 10
IMAN_MAX_CORNERS = 300
IMAN_QUALIDADE   = 0.01
IMAN_DIST_MIN    = 5
IMAN_USAR_HARRIS = True
IMAN_HARRIS_K    = 0.04

JANELA      = "Estimador de Rotacao por Quadro"
JANELA_EIXO = "Selecionar Eixo"
BAR_NAV_H   = 24   # altura da barra de navegação (scrubber clicável)
BAR_CAP_H   = 16   # altura da barra de capturas estimadas

# X=vermelho  Y=verde  Z=azul  (BGR)
CORES_EIXO_BGR = [(0, 0, 220), (0, 210, 0), (210, 70, 0)]
CORES_EIXO_NOMES = ["X", "Y", "Z"]



@dataclass
class Par:
    ponto_3d: List[float]
    ponto_2d: List[float]
    label:    int

@dataclass
class CapturaPose:
    frame:        int
    timestamp_s:  float
    eixo:         str    # "X", "Y" ou "Z"
    sinal:        float  # 1.0 ou -1.0
    angulo_graus: float
    rvec:         List[float]
    tvec:         List[float]
    euler_graus:  List[float]
    quaternion:   List[float]


CORES_BGR = [
    (80, 255, 0), (255, 200, 0), (0, 100, 255),
    (255, 0, 200), (0, 255, 200), (0, 200, 255),
    (200, 255, 0), (255, 80,  0),
]
CORES_RGB = [(r/255, g/255, b/255) for b, g, r in CORES_BGR]



def detectar_features(gray: np.ndarray) -> Optional[np.ndarray]:
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=IMAN_MAX_CORNERS, qualityLevel=IMAN_QUALIDADE,
        minDistance=IMAN_DIST_MIN, blockSize=7,
        useHarrisDetector=IMAN_USAR_HARRIS, k=IMAN_HARRIS_K)
    if corners is None:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
    return corners.reshape(-1, 2)


def snap_para_feature(pt_tela, features_tela, raio=IMAN_RAIO_PX):
    if features_tela is None or len(features_tela) == 0:
        return pt_tela, False
    diffs = features_tela - np.array(pt_tela, dtype=np.float32)
    dists = np.linalg.norm(diffs, axis=1)
    idx   = int(np.argmin(dists))
    if dists[idx] <= raio:
        p = features_tela[idx]
        return (int(round(p[0])), int(round(p[1]))), True
    return pt_tela, False


def _linha_tracejada(img, p1, p2, cor, passo=10):
    x1, y1 = p1; x2, y2 = p2
    dist = np.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    t = 0.0; ativo = True
    while t < dist:
        t2 = min(t + passo, dist)
        if ativo:
            cv2.line(img,
                     (int(x1 + dx * t),  int(y1 + dy * t)),
                     (int(x1 + dx * t2), int(y1 + dy * t2)),
                     cor, 1, cv2.LINE_AA)
        t = t2; ativo = not ativo


class EstadoCliqueIman:
    def __init__(self):
        self.pt_mouse  = None
        self.pt_clique = None
        self.pronto    = False
        self._features = None

    def reset(self):
        self.pt_clique = None
        self.pronto    = False

    def set_features(self, features_tela):
        self._features = features_tela

    def callback(self, event, x, y, flags, param):
        self.pt_mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            snapped, _ = snap_para_feature((x, y), self._features, IMAN_RAIO_PX)
            self.pt_clique = snapped
            self.pronto    = True


class EstadoNavegacao:
    def __init__(self, clique_iman: EstadoCliqueIman):
        self._iman            = clique_iman
        self.frame_solicitado = None   # frame para saltar (set on click/drag)
        self.hover_x_nav      = None   # posição X do hover na barra nav
        # Atualizados pelo loop a cada iteração
        self.largura_tela  = 1
        self.altura_tela   = 1
        self.total_frames  = 1

    @property
    def _y_nav(self):
        return self.altura_tela - BAR_NAV_H

    def _x_para_frame(self, x: int) -> int:
        x = int(np.clip(x, 0, self.largura_tela - 1))
        return int(round(x / (self.largura_tela - 1) * (self.total_frames - 1)))

    def callback(self, event, x, y, flags, param):
        in_nav = y >= self._y_nav

        if event == cv2.EVENT_MOUSEMOVE:
            self.hover_x_nav = x if in_nav else None

        if in_nav:
            # Clique ou arrasto na barra → busca o frame correspondente
            if (event == cv2.EVENT_LBUTTONDOWN or
                    (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON))):
                self.frame_solicitado = self._x_para_frame(x)
        else:
            self._iman.callback(event, x, y, flags, param)


def desenhar_features(frame, features_tela):
    if features_tela is None:
        return
    for pt in features_tela:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.line(frame, (x - 5, y), (x + 5, y), (255, 230, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (x, y - 5), (x, y + 5), (255, 230, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1, cv2.LINE_AA)


def desenhar_cursor_iman(frame, mouse_tela, features_tela):
    if mouse_tela is None:
        return
    mx, my = mouse_tela
    snapped, snappou = snap_para_feature(mouse_tela, features_tela)
    overlay = frame.copy()
    cv2.circle(overlay, (mx, my), IMAN_RAIO_PX,
               (0, 200, 255) if snappou else (120, 120, 120), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    if snappou:
        sx, sy = snapped
        _linha_tracejada(frame, (mx, my), (sx, sy), (0, 255, 255))
        cv2.circle(frame, (sx, sy), 10, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (sx, sy),  5, (255, 255,  0), 2, cv2.LINE_AA)
        cv2.circle(frame, (sx, sy),  2, (255, 255, 255), -1)
    else:
        cv2.line(frame, (mx - 8, my), (mx + 8, my), (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(frame, (mx, my - 8), (mx, my + 8), (180, 180, 180), 1, cv2.LINE_AA)



def montar_K(w: int, h: int):
    dist = np.array(DISTORCAO, dtype=np.float64)
    if METODO_K == "json":
        p = CAMINHO_K_JSON.strip()
        if not p or not os.path.exists(p):
            sys.exit(f"[ERRO] CAMINHO_K_JSON não encontrado: '{p}'")
        with open(p) as f:
            dk = json.load(f)
        return (np.array(dk["matriz_K"], dtype=np.float64),
                np.array(dk.get("distorcao", DISTORCAO), dtype=np.float64),
                dk.get("metodo", "JSON"))
    elif METODO_K == "fisico":
        fx = (FOCAL_MM / SENSOR_W_MM) * w
        fy = (FOCAL_MM / SENSOR_H_MM) * h
        K  = np.array([[fx, 0., w / 2. + CX_OFFSET],
                       [0., fy, h / 2. + CY_OFFSET],
                       [0., 0., 1.]], dtype=np.float64)
        nota = f"Física: f={FOCAL_MM}mm sensor={SENSOR_W_MM}×{SENSOR_H_MM}mm"
    elif METODO_K == "fov":
        fx = (w / 2.) / np.tan(np.deg2rad(FOV_H_GRAUS) / 2.)
        K  = np.array([[fx, 0., w / 2.], [0., fx, h / 2.], [0., 0., 1.]], dtype=np.float64)
        nota = f"FOV={FOV_H_GRAUS}°"
    else:
        fx = float(max(w, h))
        K  = np.array([[fx, 0., w / 2.], [0., fx, h / 2.], [0., 0., 1.]], dtype=np.float64)
        nota = f"Heurística f={fx:.0f}px"
    return K, dist, nota



def _esfera(centro, cor_rgb, raio=0.008):
    e = o3d.geometry.TriangleMesh.create_sphere(radius=raio)
    e.translate(centro)
    e.paint_uniform_color(list(cor_rgb))
    e.compute_vertex_normals()
    return e


def salvar_camera3d(vis, d):
    try:
        ctrl = vis.get_view_control()
        p    = ctrl.convert_to_pinhole_camera_parameters()
        data = {
            "extrinsic": p.extrinsic.tolist(),
            "intrinsic": {
                "width":  p.intrinsic.width,
                "height": p.intrinsic.height,
                "intrinsic_matrix": p.intrinsic.intrinsic_matrix.tolist(),
            },
        }
        with open(os.path.join(d, "camera3d.json"), "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"    [AVISO] Não salvou câmera 3D: {e}")


def restaurar_camera3d(vis, d):
    p = os.path.join(d, "camera3d.json")
    if not os.path.exists(p):
        return
    try:
        data  = json.load(open(p))
        ctrl  = vis.get_view_control()
        par   = ctrl.convert_to_pinhole_camera_parameters()
        par.extrinsic = np.array(data["extrinsic"])
        idata = data["intrinsic"]
        par.intrinsic.set_intrinsics(
            idata["width"], idata["height"],
            idata["intrinsic_matrix"][0][0],
            idata["intrinsic_matrix"][1][1],
            idata["intrinsic_matrix"][0][2],
            idata["intrinsic_matrix"][1][2])
        ctrl.convert_from_pinhole_camera_parameters(par, allow_arbitrary=True)
    except Exception as e:
        print(f"    [AVISO] Não restaurou câmera 3D: {e}")


def selecionar_ponto_3d(mesh, label: int, pares: List[Par], script_dir: str):
    if SUBDIVIDIR_MALHA:
        mesh_sel = o3d.geometry.TriangleMesh(mesh)
        mesh_sel = (mesh_sel.subdivide_loop(SUBDIVISAO_ITERACOES)
                    if SUBDIVISAO_METODO == "loop"
                    else mesh_sel.subdivide_midpoint(SUBDIVISAO_ITERACOES))
        mesh_sel.compute_vertex_normals()
    else:
        mesh_sel = mesh

    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(
        window_name=f"Ponto {label} — SHIFT+Clique no vértice | feche para confirmar",
        width=1100, height=750)
    vis.add_geometry(mesh_sel)
    for par in pares:
        vis.add_geometry(_esfera(par.ponto_3d, CORES_RGB[(par.label - 1) % len(CORES_RGB)]))
    vis.poll_events(); vis.update_renderer()
    restaurar_camera3d(vis, script_dir)
    vis.poll_events(); vis.update_renderer()
    print(f"    [3D] SHIFT+Clique no vértice do Ponto {label} e feche a janela.")
    vis.run()
    salvar_camera3d(vis, script_dir)
    vis.destroy_window()

    picked = vis.get_picked_points()
    if not picked:
        print("    [AVISO] Nenhum vértice selecionado.")
        return None
    coord = list(picked[0].coord)
    print(f"    [OK] Ponto {label} 3D: {[f'{v:.4f}' for v in coord]}")
    # Open3D usa Y-up; converte para espaço OpenCV: [x, -y, -z]
    return [coord[0], -coord[1], -coord[2]]



def escala_para_tela(frame, max_w=1280, max_h=700):
    h, w   = frame.shape[:2]
    escala = min(max_w / w, max_h / h, 1.0)
    if escala < 1.0:
        frame = cv2.resize(frame, (int(w * escala), int(h * escala)),
                           interpolation=cv2.INTER_AREA)
    return frame, escala


def tela_para_orig(pt, escala):
    return [pt[0] / escala, pt[1] / escala]


def orig_para_tela(pt, escala):
    return (int(pt[0] * escala), int(pt[1] * escala))



def calcular_pose(pares: List[Par], K, dist):
    if len(pares) < 4:
        return None
    pts3d = np.array([p.ponto_3d for p in pares], dtype=np.float64)
    pts2d = np.array([p.ponto_2d for p in pares], dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, dist,
        iterationsCount=300, reprojectionError=8.0, confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    if inliers is not None and len(inliers) >= 4:
        idx = inliers.ravel()
        cv2.solvePnP(pts3d[idx], pts2d[idx], K, dist, rvec, tvec,
                     useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    return rvec, tvec


def erro_reproj_medio(pares, K, dist, rvec, tvec) -> float:
    pts3d = np.array([p.ponto_3d for p in pares], dtype=np.float64)
    pts2d = np.array([p.ponto_2d for p in pares], dtype=np.float64)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    return float(np.mean(np.linalg.norm(pts2d - proj.reshape(-1, 2), axis=1)))



def _txt(frame, txt, pos, cor=(255, 220, 50), sf=0.50):
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, sf, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, sf, cor,       1, cv2.LINE_AA)


def cor_par(label):
    return CORES_BGR[(label - 1) % len(CORES_BGR)]


def desenhar_pares(frame, pares, escala):
    for par in pares:
        pt = orig_para_tela(par.ponto_2d, escala)
        c  = cor_par(par.label)
        cv2.circle(frame, pt, 7, c, -1)
        cv2.circle(frame, pt, 8, (255, 255, 255), 1)
        cv2.putText(frame, str(par.label), (pt[0] + 10, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, str(par.label), (pt[0] + 10, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)


def desenhar_eixos_3d(frame, K, dist, rvec, tvec, escala, comp=0.1):
    pts  = np.float32([[0, 0, 0], [comp, 0, 0], [0, comp, 0], [0, 0, comp]])
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    proj = (proj.reshape(-1, 2) * escala).astype(int)
    o    = tuple(proj[0])
    for i, cor in enumerate(CORES_EIXO_BGR):
        cv2.arrowedLine(frame, o, tuple(proj[i + 1]), cor, 2, tipLength=0.2, line_type=cv2.LINE_AA)


def desenhar_eixos_rotulados(frame, K, dist, rvec, tvec, escala,
                              eixo_sel: int, comp: float = 0.12):
    pts  = np.float32([[0, 0, 0], [comp, 0, 0], [0, comp, 0], [0, 0, comp]])
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    proj = (proj.reshape(-1, 2) * escala).astype(int)
    o    = tuple(proj[0])

    for i, (cor, nome) in enumerate(zip(CORES_EIXO_BGR, CORES_EIXO_NOMES)):
        sel       = (i == eixo_sel)
        espessura = 4 if sel else 2
        p         = tuple(proj[i + 1])

        cv2.arrowedLine(frame, o, p, cor, espessura, tipLength=0.25, line_type=cv2.LINE_AA)

        lx, ly = p[0] + 8, p[1] + 5
        sf = 0.80 if sel else 0.60
        cv2.putText(frame, nome, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, sf,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, nome, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, sf,
                    cor, 2, cv2.LINE_AA)

        if sel:
            cv2.circle(frame, p, 7, cor, 2, cv2.LINE_AA)
            cv2.circle(frame, p, 4, cor, -1, cv2.LINE_AA)


def desenhar_timeline(frame, capturas: List[CapturaPose],
                      frame_atual: int, total_frames: int,
                      hover_x: Optional[int] = None):
    h, w     = frame.shape[:2]
    y_cap    = h - BAR_NAV_H - BAR_CAP_H   # topo da barra de capturas
    y_nav    = h - BAR_NAV_H               # topo da barra de navegação

    # Fundos
    cv2.rectangle(frame, (0, y_cap), (w, y_nav), (28, 28, 28), -1)
    cv2.rectangle(frame, (0, y_nav), (w,    h  ), (42, 42, 42), -1)
    cv2.line(frame, (0, y_nav), (w, y_nav), (65, 65, 65), 1)

    # Labels
    cv2.putText(frame, "EST", (4, y_nav - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (60, 140, 60), 1, cv2.LINE_AA)
    cv2.putText(frame, "NAV", (4, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (60, 130, 150), 1, cv2.LINE_AA)

    if total_frames <= 1:
        return

    def f2x(f):
        return int(f / (total_frames - 1) * (w - 1))

    for cap in capturas:
        x   = f2x(cap.frame)
        cor = (0, 255, 255) if cap.frame == frame_atual else (0, 210, 80)
        cv2.rectangle(frame, (x - 2, y_cap + 2), (x + 2, y_nav - 2), cor, -1)

    xc = f2x(frame_atual)
    cv2.rectangle(frame, (0, y_nav + 1), (xc, h - 1), (35, 85, 105), -1)

    if hover_x is not None:
        hf = int(round(hover_x / (w - 1) * (total_frames - 1)))
        cv2.line(frame, (hover_x, y_nav), (hover_x, h), (130, 130, 130), 1)
        txt = str(hf)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tx = int(np.clip(hover_x - tw // 2, 2, w - tw - 2))
        cv2.putText(frame, txt, (tx, y_nav - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)

    cv2.line(frame, (xc, y_nav), (xc, h), (0, 215, 255), 2)
    cy_nav = y_nav + BAR_NAV_H // 2
    cv2.circle(frame, (xc, cy_nav), 6, (0, 215, 255), -1)
    cv2.circle(frame, (xc, cy_nav), 6, (255, 255, 255), 1)



def hud_navegacao(frame, frame_atual: int, total_frames: int,
                  ts: float, capturas: List[CapturaPose]):
    cap_aqui = next((c for c in capturas if c.frame == frame_atual), None)
    linhas = [
        f"Frame {frame_atual}/{total_frames - 1}   t={ts:.3f}s   Capturas={len(capturas)}",
        "[A] Capturar   [D] Deletar   [L] Listar   [Q] Salvar e sair",
    ]
    if cap_aqui:
        linhas.append(
            f">>> Captura aqui: eixo {cap_aqui.eixo}  "
            f"{'(invertido)' if cap_aqui.sinal < 0 else ''}  "
            f"= {cap_aqui.angulo_graus:.2f}°")
    for i, txt in enumerate(linhas):
        cor = (0, 255, 120) if i == 2 else (255, 220, 50)
        _txt(frame, txt, (8, 24 + i * 24), cor)


def hud_captura(frame, frame_atual: int, pares: List[Par],
                instrucao: str, erro: Optional[float] = None):
    err_txt = f"   Reproj={erro:.1f}px" if erro is not None else ""
    linhas  = [
        f"[CAPTURA] Frame {frame_atual}   Pares={len(pares)}{err_txt}",
        "[A] Add par   [D] Del par   [R] Reset   [ESPACO] Calcular   [ESC] Cancelar",
    ]
    if instrucao:
        linhas.append(instrucao)
    for i, txt in enumerate(linhas):
        cor = (0, 220, 255) if i == 2 else (255, 140, 0)
        _txt(frame, txt, (8, 24 + i * 24), cor)



def _painel_eixo(euler_graus: List[float], sel: int, sinal: float) -> np.ndarray:
    W, H = 420, 330
    p    = np.full((H, W, 3), (22, 22, 22), dtype=np.uint8)

    _txt(p, "SELECIONAR EIXO DE ROTACAO", (18, 36), (230, 230, 230), 0.62)
    cv2.line(p, (18, 50), (W - 18, 50), (60, 60, 60), 1)

    for i, (cor, nome, ang) in enumerate(zip(CORES_EIXO_BGR, CORES_EIXO_NOMES, euler_graus)):
        y0_bg = 60 + i * 58
        y_txt = y0_bg + 34

        if i == sel:
            cv2.rectangle(p, (14, y0_bg + 6), (W - 14, y0_bg + 52), (40, 40, 50), -1)
            cv2.rectangle(p, (14, y0_bg + 6), (W - 14, y0_bg + 52), cor, 1)

        cv2.rectangle(p, (22, y_txt - 14), (42, y_txt + 4), cor, -1)

        cor_txt = (255, 255, 255) if i == sel else (140, 140, 140)
        seta    = ">> " if i == sel else "   "
        _txt(p, f"{seta}[{nome}]  {ang:+.2f} graus", (50, y_txt), cor_txt, 0.62)

    y_sep = 60 + 3 * 58 + 8
    cv2.line(p, (18, y_sep), (W - 18, y_sep), (60, 60, 60), 1)

    sinal_txt    = "+" if sinal > 0 else "-"
    angulo_final = float(euler_graus[sel]) * sinal
    _txt(p, f"Sinal atual: {sinal_txt}   |   [S] para inverter",
         (18, y_sep + 28), (180, 180, 180), 0.50)
    _txt(p, f"Resultado: {angulo_final:+.2f} graus",
         (18, y_sep + 56), (0, 215, 110), 0.60)

    cv2.line(p, (18, H - 40), (W - 18, H - 40), (45, 45, 45), 1)
    _txt(p, "[X/Y/Z] eixo   [S] sinal   [ENTER] OK   [ESC] cancelar",
         (18, H - 16), (100, 100, 100), 0.40)

    return p


def selecionar_eixo(frame_orig, euler_graus: List[float],
                    K, dist, rvec, tvec) -> Optional[Tuple[str, float, float]]:
    sel   = 1      # Y por padrão
    sinal = 1.0
    comp  = max(float(np.linalg.norm(tvec)) * 0.15, 0.05)

    cv2.namedWindow(JANELA_EIXO, cv2.WINDOW_NORMAL)

    while True:
        exib, escala = escala_para_tela(frame_orig.copy())
        desenhar_eixos_rotulados(exib, K, dist, rvec, tvec, escala, sel, comp)
        _txt(exib, "Veja o painel 'Selecionar Eixo'",
             (8, exib.shape[0] - BAR_NAV_H - BAR_CAP_H - 8), (180, 180, 180), 0.45)
        cv2.imshow(JANELA, exib)

        cv2.imshow(JANELA_EIXO, _painel_eixo(euler_graus, sel, sinal))

        key = cv2.waitKey(40) & 0xFF

        if   key == ord('x'): sel = 0
        elif key == ord('y'): sel = 1
        elif key == ord('z'): sel = 2
        elif key == ord('s'): sinal = -sinal
        elif key in (13, 10):   # ENTER
            cv2.destroyWindow(JANELA_EIXO)
            return CORES_EIXO_NOMES[sel], sinal, float(euler_graus[sel]) * sinal
        elif key == 27:          # ESC
            cv2.destroyWindow(JANELA_EIXO)
            return None



def carregar_capturas(caminho: str) -> List[CapturaPose]:
    if not os.path.exists(caminho):
        return []
    try:
        with open(caminho) as f:
            raw = json.load(f)
        capturas = [CapturaPose(**r) for r in raw]
        capturas.sort(key=lambda c: c.frame)
        print(f"[OK] {len(capturas)} capturas carregadas de: {caminho}")
        return capturas
    except Exception as e:
        print(f"[AVISO] Não foi possível carregar capturas: {e}")
        return []


def salvar_capturas(capturas: List[CapturaPose], caminho: str):
    ordenadas = sorted(capturas, key=lambda c: c.frame)
    with open(caminho, "w") as f:
        json.dump([asdict(c) for c in ordenadas], f, indent=2)



def modo_captura(frame_orig, frame_atual: int, fps: float,
                 K, dist, mesh, script_dir: str,
                 clique: EstadoCliqueIman) -> Optional[CapturaPose]:
    pares:         List[Par] = []
    proximo_label  = 1
    ult_rvec       = None
    ult_tvec       = None
    ult_erro       = None
    instrucao      = "Adicione pares com [A] (min. 4) | [ESPACO] para calcular"
    modo_iman      = False
    feats_cache    = None

    print(f"\n  [CAPTURA] Frame {frame_atual}")

    while True:
        exib, escala = escala_para_tela(frame_orig.copy())

        if ult_rvec is not None:
            comp = float(np.linalg.norm(ult_tvec)) * 0.15 or 0.05
            desenhar_eixos_3d(exib, K, dist, ult_rvec, ult_tvec, escala, comp)

        desenhar_pares(exib, pares, escala)

        if modo_iman:
            desenhar_features(exib, feats_cache)
            desenhar_cursor_iman(exib, clique.pt_mouse, feats_cache)

        hud_captura(exib, frame_atual, pares, instrucao, ult_erro)
        cv2.imshow(JANELA, exib)

        key_ex = cv2.waitKeyEx(16 if modo_iman else 40)
        key    = key_ex & 0xFF

        if key == 27:
            print("  [CAPTURA] Cancelada.")
            return None

        elif key == ord('r'):
            pares.clear()
            ult_rvec = ult_tvec = ult_erro = None
            proximo_label = 1
            instrucao = "Pares resetados."
            print("  [R] Pares resetados.")

        elif key == ord('d') and not modo_iman:
            if not pares:
                instrucao = "Nenhum par para deletar."
                continue
            instrucao = "Clique perto do ponto para deletar  |  [ESC] Cancelar"
            clique.reset()
            cancelado = False
            while not clique.pronto:
                exib2, esc2 = escala_para_tela(frame_orig.copy())
                desenhar_pares(exib2, pares, esc2)
                hud_captura(exib2, frame_atual, pares, instrucao, ult_erro)
                cv2.imshow(JANELA, exib2)
                if cv2.waitKey(30) & 0xFF == 27:
                    cancelado = True
                    break
            if not cancelado and clique.pronto:
                mx, my = clique.pt_clique
                dists = [np.hypot(*(np.array(orig_para_tela(p.ponto_2d, escala)) - [mx, my]))
                         for p in pares]
                idx = int(np.argmin(dists))
                if dists[idx] <= 40:
                    rem = pares.pop(idx)
                    instrucao = f"Par {rem.label} removido."
                    print(f"  [OK] Par {rem.label} removido.")
                else:
                    instrucao = "Nenhum ponto próximo."
            else:
                instrucao = "Deleção cancelada."
            clique.reset()

        elif key == ord(' '):
            if len(pares) < 4:
                instrucao = f"Adicione mais {4 - len(pares)} par(es) antes de calcular."
                continue
            res = calcular_pose(pares, K, dist)
            if res is None:
                instrucao = "solvePnP falhou. Verifique os pares e tente novamente."
                print("  [AVISO] solvePnP falhou.")
                continue
            ult_rvec, ult_tvec = res
            ult_erro = erro_reproj_medio(pares, K, dist, ult_rvec, ult_tvec)
            rot      = Rotation.from_matrix(cv2.Rodrigues(ult_rvec)[0])
            euler    = rot.as_euler("xyz", degrees=True).tolist()
            quat     = rot.as_quat().tolist()
            print(f"  [POSE] Reproj={ult_erro:.2f}px  "
                  f"euler=[{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]°")

            res_eixo = selecionar_eixo(frame_orig, euler, K, dist, ult_rvec, ult_tvec)
            if res_eixo is None:
                instrucao = "Seleção de eixo cancelada. Recalcule com [ESPACO]."
                ult_rvec = ult_tvec = None
                continue

            eixo_str, sinal, angulo = res_eixo
            return CapturaPose(
                frame=frame_atual,
                timestamp_s=round(frame_atual / fps, 4),
                eixo=eixo_str,
                sinal=sinal,
                angulo_graus=round(angulo, 3),
                rvec=[round(v, 6) for v in ult_rvec.ravel().tolist()],
                tvec=[round(v, 6) for v in ult_tvec.ravel().tolist()],
                euler_graus=[round(e, 3) for e in euler],
                quaternion=[round(q, 6) for q in quat],
            )

        elif key == ord('a'):
            label      = proximo_label
            modo_iman  = True
            gray       = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
            feats_orig = detectar_features(gray)
            _, escala  = escala_para_tela(frame_orig)
            feats_cache = feats_orig * escala if feats_orig is not None else None
            clique.set_features(feats_cache)
            clique.reset()
            n_feats   = len(feats_cache) if feats_cache is not None else 0
            instrucao = f"Ponto {label}: clique no vídeo  [{n_feats} features]  ESC=cancelar"
            print(f"  [A] Ponto {label} — clique no vídeo...")

            cancelado = False
            while not clique.pronto:
                exib2, esc2 = escala_para_tela(frame_orig.copy())
                desenhar_pares(exib2, pares, esc2)
                desenhar_features(exib2, feats_cache)
                desenhar_cursor_iman(exib2, clique.pt_mouse, feats_cache)
                hud_captura(exib2, frame_atual, pares, instrucao, ult_erro)
                cv2.imshow(JANELA, exib2)
                if cv2.waitKey(16) & 0xFF == 27:
                    cancelado = True
                    break

            modo_iman = False
            clique.set_features(None)

            if cancelado or not clique.pronto:
                instrucao = "Adição cancelada."
                clique.reset()
                continue

            pt_tela = clique.pt_clique
            _, esc  = escala_para_tela(frame_orig)
            pt_orig = tela_para_orig(pt_tela, esc)
            clique.reset()

            exib3, esc3 = escala_para_tela(frame_orig.copy())
            desenhar_pares(exib3, pares, esc3)
            cv2.circle(exib3, pt_tela, 11, (0, 255, 255), 2)
            cv2.circle(exib3, pt_tela,  7, cor_par(label), -1)
            hud_captura(exib3, frame_atual, pares,
                        f"Ponto {label} marcado — selecione vértice no modelo 3D...")
            cv2.imshow(JANELA, exib3)
            cv2.waitKey(200)

            pt3d = selecionar_ponto_3d(mesh, label, pares, script_dir)
            if pt3d is None:
                instrucao = f"Ponto {label} cancelado."
                continue

            pares.append(Par(ponto_3d=pt3d, ponto_2d=pt_orig, label=label))
            proximo_label += 1
            faltam = max(0, 4 - len(pares))
            instrucao = (f"Par {label} adicionado! Total: {len(pares)}. " +
                         (f"Faltam {faltam}." if faltam else "Pronto — [ESPACO] para calcular."))
            print(f"  [OK] Par {label}. Total: {len(pares)}.")



def loop_principal(caminho_video: str, K, dist, mesh,
                   caminho_json: str, script_dir: str):

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        sys.exit(f"[ERRO] Não foi possível abrir: {caminho_video}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)

    capturas: List[CapturaPose] = carregar_capturas(caminho_json)

    clique = EstadoCliqueIman()
    nav    = EstadoNavegacao(clique)
    cv2.setMouseCallback(JANELA, nav.callback)

    frame_atual = 0
    frame_orig  = None

    KC_DIR = 0x270000
    KC_ESQ = 0x250000
    ultima_seta = None
    repeticoes  = 0

    def ler_frame(n: int):
        n = max(0, min(total_frames - 1, n))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        r, f = cap.read()
        return f if r else None

    frame_orig = ler_frame(0)

    print(f"\n[NAVEGACAO] {total_frames} frames @ {fps:.1f} fps")
    print("  Setas: navegar   [A] Capturar   [D] Deletar   [L] Listar   [Q] Sair\n")

    while True:
        if frame_orig is None:
            frame_orig = ler_frame(frame_atual)
            if frame_orig is None:
                break

        exib, escala_tela = escala_para_tela(frame_orig.copy())
        ts = frame_atual / fps

        nav.largura_tela  = exib.shape[1]
        nav.altura_tela   = exib.shape[0]
        nav.total_frames  = total_frames

        desenhar_timeline(exib, capturas, frame_atual, total_frames, nav.hover_x_nav)
        hud_navegacao(exib, frame_atual, total_frames, ts, capturas)
        cv2.imshow(JANELA, exib)

        key_ex = cv2.waitKeyEx(40)
        key    = key_ex & 0xFF

        if nav.frame_solicitado is not None:
            novo = nav.frame_solicitado
            nav.frame_solicitado = None
            if novo != frame_atual:
                frame_atual = novo
                frame_orig  = ler_frame(frame_atual)
            continue

        if key == ord('q'):
            break

        elif key_ex in (KC_DIR, KC_ESQ):
            sinal_nav   = +1 if key_ex == KC_DIR else -1
            repeticoes  = (repeticoes + 1) if key_ex == ultima_seta else 0
            ultima_seta = key_ex
            # 1 frame no primeiro toque, sobe até 100 ao segurar
            delta = min(1 + (repeticoes // 4) * 10, 100)
            novo  = max(0, min(total_frames - 1, frame_atual + sinal_nav * delta))
            if novo != frame_atual:
                frame_atual = novo
                frame_orig  = ler_frame(frame_atual)
            continue
        else:
            ultima_seta = None
            repeticoes  = 0

        if key == ord('d'):
            antes     = len(capturas)
            capturas  = [c for c in capturas if c.frame != frame_atual]
            if len(capturas) < antes:
                salvar_capturas(capturas, caminho_json)
                print(f"  [OK] Captura no frame {frame_atual} deletada.")
            else:
                print(f"  [AVISO] Nenhuma captura no frame {frame_atual}.")

        elif key == ord('l'):
            if not capturas:
                print("  [CAPTURAS] Nenhuma captura registrada.")
            else:
                print(f"\n  [CAPTURAS] {len(capturas)} total (ordem cronológica):")
                for c in capturas:
                    inv = " (invertido)" if c.sinal < 0 else ""
                    print(f"    Frame {c.frame:5d}  t={c.timestamp_s:.3f}s  "
                          f"eixo={c.eixo}{inv}  angulo={c.angulo_graus:.2f}°")
                print()

        elif key == ord('a'):
            existente = next((c for c in capturas if c.frame == frame_atual), None)
            if existente:
                capturas = [c for c in capturas if c.frame != frame_atual]
                print(f"  [CAPTURA] Reestimando frame {frame_atual}...")

            resultado = modo_captura(
                frame_orig, frame_atual, fps,
                K, dist, mesh, script_dir, clique)

            cv2.setMouseCallback(JANELA, nav.callback)

            if resultado is not None:
                capturas.append(resultado)
                capturas.sort(key=lambda c: c.frame)
                salvar_capturas(capturas, caminho_json)
                print(f"  [SALVO] frame={frame_atual}  "
                      f"eixo={resultado.eixo}  "
                      f"angulo={resultado.angulo_graus:.2f}°")

    cap.release()
    cv2.destroyAllWindows()
    salvar_capturas(capturas, caminho_json)
    print(f"\n[FIM] {len(capturas)} capturas salvas em:\n  {caminho_json}")



_tk_root = None

def _get_root():
    global _tk_root
    if _tk_root is None:
        _tk_root = tk.Tk()
        _tk_root.withdraw()
    return _tk_root

def pedir_arquivo(titulo, tipos):
    r = _get_root(); r.lift(); r.focus_force()
    return filedialog.askopenfilename(parent=r, title=titulo, filetypes=tipos)

def encontrar_no_dir(diretorio, extensoes):
    for f in sorted(os.listdir(diretorio)):
        if any(f.lower().endswith(e) for e in extensoes):
            return os.path.join(diretorio, f)
    return None



if __name__ == "__main__":
    print("=== Estimador de Rotação por Quadro ===")
    print(f"    Método K : {METODO_K}")
    if METODO_K == "fisico":
        print(f"    focal={FOCAL_MM}mm  sensor={SENSOR_W_MM}×{SENSOR_H_MM}mm")
    print()

    aqui = os.path.dirname(os.path.abspath(__file__))

    # Selecionar vídeo
    v = pedir_arquivo("Selecione o vídeo",
                      [("Vídeos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Todos", "*.*")])
    if not v:
        sys.exit("Cancelado.")
    print(f"[OK] Vídeo: {v}")

    # JSON na mesma pasta do vídeo
    pasta_video  = os.path.dirname(v)
    nome_base    = os.path.splitext(os.path.basename(v))[0]
    caminho_json = os.path.join(pasta_video, nome_base + "_rotacao.json")
    print(f"[OK] JSON  : {caminho_json}")

    # Dimensões para montar K
    _cap = cv2.VideoCapture(v)
    _w   = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _h   = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _cap.release()

    K, dist, nota_K = montar_K(_w, _h)
    print(f"[OK] K     : {nota_K}")

    # Modelo 3D
    obj_path = encontrar_no_dir(aqui, [".obj"])
    if obj_path:
        print(f"[AUTO] Modelo: {obj_path}")
    else:
        obj_path = pedir_arquivo("Selecione o modelo .obj",
                                 [("OBJ", "*.obj"), ("Todos", "*.*")])
        if not obj_path:
            sys.exit("Cancelado.")
    print(f"[OK] Carregando modelo 3D...")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    print(f"[OK] Modelo carregado ({len(np.asarray(mesh.vertices))} vértices).\n")

    loop_principal(v, K, dist, mesh, caminho_json, aqui)
