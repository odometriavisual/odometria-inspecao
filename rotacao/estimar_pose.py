import cv2
import numpy as np
import open3d as o3d
import json
import os
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

# CONFIGURAÇÕES

# Método de estimação de K:
#   "fisico"     → usa FOCAL_MM + SENSOR_W_MM + SENSOR_H_MM
#   "fov"        → usa FOV_H_GRAUS
#   "heuristica" → f = max(largura, altura)  [menos preciso]
#   "json"       → carrega de CAMINHO_K_JSON
METODO_K = "fisico"

# Parâmetros físicos da câmera (usados quando METODO_K = "fisico")
FOCAL_MM    = 4.25
SENSOR_W_MM = 6.17
SENSOR_H_MM = 4.55
CX_OFFSET   = 0.0
CY_OFFSET   = 0.0

FOV_H_GRAUS = 90.0
CAMINHO_K_JSON = ""

USAR_PREVIEW_3D = False
DISTORCAO = [0.0, 0.0, 0.0, 0.0, 0.0]

# Imã: raio em pixels (na imagem exibida) dentro do qual o cursor snapeia
IMAN_RAIO_PX     = 10
# Quantos features detectar no frame para o imã
IMAN_MAX_CORNERS = 300
# Qualidade mínima do feature (0.0–1.0)
# Valores menores = mais features, mais bordas fracas incluídas
IMAN_QUALIDADE   = 0.01
# Distância mínima entre features detectados (px) — menor = mais denso
IMAN_DIST_MIN    = 5
# True  → Harris corner detector (favorece cantos/arestas do modelo)
# False → mínimo autovalor (detecta mais regiões planas também)
IMAN_USAR_HARRIS = True
IMAN_HARRIS_K    = 0.04   # parâmetro Harris (0.04–0.06 é o usual)
# subdivisão da malha para ter mais vértices selecionáveis
SUBDIVIDIR_MALHA        = True
SUBDIVISAO_ITERACOES    = 2   # 1=dobra vértices, 2=quadruplica, 3=octuplica
# "midpoint" é mais rápido; "loop" é mais suave mas mais lento
SUBDIVISAO_METODO       = "midpoint"  # "midpoint" ou "loop"


@dataclass
class Par:
    ponto_3d: List[float]
    ponto_2d: List[float]
    label: int

@dataclass
class RegistroPose:
    timestamp_s: float
    numero_frame: int
    rvec: List[float]
    tvec: List[float]
    quaternion: List[float]
    euler_graus: List[float]
    sucesso: bool
    num_pontos: int

CORES_BGR = [
    (80, 255, 0), (255, 200, 0), (0, 100, 255),
    (255, 0, 200), (0, 255, 200), (0, 200, 255),
    (200, 255, 0), (255, 80,  0),
]
CORES_RGB = [(r/255, g/255, b/255) for b, g, r in CORES_BGR]

_R_CV2O3D = np.array([
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0, -1],
], dtype=np.float64)
# Detecção de features para o imã
def detectar_features(gray: np.ndarray) -> Optional[np.ndarray]:
    """Detecta goodFeaturesToTrack e retorna array (N,2) ou None.
    Com IMAN_USAR_HARRIS=True usa Harris corner detector, que favorece
    cantos reais (bordas do modelo) em vez de regiões planas."""
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=IMAN_MAX_CORNERS,
        qualityLevel=IMAN_QUALIDADE,
        minDistance=IMAN_DIST_MIN,
        blockSize=7,
        useHarrisDetector=IMAN_USAR_HARRIS,
        k=IMAN_HARRIS_K if IMAN_USAR_HARRIS else 0.04,
    )
    if corners is None:
        return None
    # Refina com sub-pixel
    corners = cv2.cornerSubPix(
        gray, corners, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
    )
    return corners.reshape(-1, 2)  # (N, 2)

def snap_para_feature(
    pt_tela: Tuple[int, int],
    features_tela: Optional[np.ndarray],
    raio: int = IMAN_RAIO_PX,
) -> Tuple[Tuple[int, int], bool]:
    """
    Retorna (ponto_snappeado, snappou).
    Se não houver feature dentro do raio, retorna o ponto original.
    """
    if features_tela is None or len(features_tela) == 0:
        return pt_tela, False
    diffs = features_tela - np.array(pt_tela, dtype=np.float32)
    dists = np.linalg.norm(diffs, axis=1)
    idx   = int(np.argmin(dists))
    if dists[idx] <= raio:
        p = features_tela[idx]
        return (int(round(p[0])), int(round(p[1]))), True
    return pt_tela, False
# Desenho de features (overlay do imã)
def desenhar_features(frame: np.ndarray, features_tela: Optional[np.ndarray]):
    """Desenha cruzes ciano nos pontos detectados."""
    if features_tela is None:
        return
    for pt in features_tela:
        x, y = int(round(pt[0])), int(round(pt[1]))
        # Cruz pequena
        cv2.line(frame, (x-5, y), (x+5, y), (255, 230, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (x, y-5), (x, y+5), (255, 230, 0), 1, cv2.LINE_AA)
        # Ponto central
        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1, cv2.LINE_AA)

def desenhar_cursor_iman(
    frame: np.ndarray,
    mouse_tela: Optional[Tuple[int, int]],
    features_tela: Optional[np.ndarray],
    raio: int = IMAN_RAIO_PX,
):
    """
    Desenha:
      - Círculo de atração ao redor do cursor (zona do imã)
      - Linha pontilhada do cursor até o feature mais próximo (se dentro do raio)
      - Highlight no feature alvo
    """
    if mouse_tela is None:
        return

    mx, my = mouse_tela
    snapped_pt, snappou = snap_para_feature(mouse_tela, features_tela, raio)

    # Zona de atração — círculo semitransparente
    overlay = frame.copy()
    cor_zona = (0, 200, 255) if snappou else (120, 120, 120)
    cv2.circle(overlay, (mx, my), raio, cor_zona, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    if snappou:
        sx, sy = snapped_pt
        # Linha tracejada do cursor ao feature
        _linha_tracejada(frame, (mx, my), (sx, sy), (0, 255, 255), passo=8)
        # Highlight no feature alvo: círculo pulsante simulado com dois anéis
        cv2.circle(frame, (sx, sy), 10, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (sx, sy),  5, (255, 255,  0), 2, cv2.LINE_AA)
        cv2.circle(frame, (sx, sy),  2, (255, 255, 255), -1)
    else:
        # Cursor simples (mira)
        cv2.line(frame, (mx-8, my), (mx+8, my), (180, 180, 180), 1, cv2.LINE_AA)
        cv2.line(frame, (mx, my-8), (mx, my+8), (180, 180, 180), 1, cv2.LINE_AA)

def _linha_tracejada(img, p1, p2, cor, passo=10, espessura=1):
    """Desenha linha tracejada entre dois pontos."""
    x1, y1 = p1; x2, y2 = p2
    dist = np.hypot(x2-x1, y2-y1)
    if dist < 1: return
    dx, dy = (x2-x1)/dist, (y2-y1)/dist
    t = 0.0
    ativo = True
    while t < dist:
        t2 = min(t + passo, dist)
        if ativo:
            pa = (int(x1 + dx*t),  int(y1 + dy*t))
            pb = (int(x1 + dx*t2), int(y1 + dy*t2))
            cv2.line(img, pa, pb, cor, espessura, cv2.LINE_AA)
        t    = t2
        ativo = not ativo
# Estado do mouse com imã
class EstadoCliqueIman:
    """
    Mouse callback que:
      - Rastreia posição do cursor em tempo real
      - No clique, snapeia para o goodFeature mais próximo
    """
    def __init__(self):
        self.pt_mouse:   Optional[Tuple[int,int]] = None  # posição atual (tela)
        self.pt_clique:  Optional[Tuple[int,int]] = None  # posição do clique (tela, snappada)
        self.pronto:     bool = False
        self._features:  Optional[np.ndarray] = None      # features em coord de tela

    def reset(self):
        self.pt_clique = None
        self.pronto    = False

    def set_features(self, features_tela: Optional[np.ndarray]):
        self._features = features_tela

    def callback(self, event, x, y, flags, param):
        self.pt_mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            snapped, _ = snap_para_feature((x, y), self._features, IMAN_RAIO_PX)
            self.pt_clique = snapped
            self.pronto    = True
# Estimação de K
def K_heuristica(largura: int, altura: int) -> Tuple[np.ndarray, str]:
    fx = float(max(largura, altura))
    K = np.array([[fx, 0., largura/2.], [0., fx, altura/2.], [0., 0., 1.]], dtype=np.float64)
    return K, f"Heurística (f=max(w,h)={fx:.0f}px)"

def K_por_parametros_fisicos(largura: int, altura: int) -> Tuple[np.ndarray, str]:
    fx = (FOCAL_MM / SENSOR_W_MM) * largura
    fy = (FOCAL_MM / SENSOR_H_MM) * altura
    cx = largura / 2. + CX_OFFSET
    cy = altura  / 2. + CY_OFFSET
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]], dtype=np.float64)
    nota = (f"Física: f={FOCAL_MM}mm sensor={SENSOR_W_MM}×{SENSOR_H_MM}mm "
            f"→ fx={fx:.1f} fy={fy:.1f}px")
    return K, nota

def K_por_fov(largura: int, altura: int) -> Tuple[np.ndarray, str]:
    fov_rad = np.deg2rad(FOV_H_GRAUS)
    fx = (largura / 2.) / np.tan(fov_rad / 2.)
    K = np.array([[fx, 0., largura/2.], [0., fx, altura/2.], [0., 0., 1.]], dtype=np.float64)
    return K, f"FOV horizontal={FOV_H_GRAUS}° → fx=fy={fx:.1f}px"

def K_de_json(largura: int, altura: int) -> Tuple[np.ndarray, np.ndarray, str]:
    p = CAMINHO_K_JSON.strip()
    if not p or not os.path.exists(p):
        sys.exit(f"[ERRO] CAMINHO_K_JSON não encontrado: '{p}'")
    with open(p) as f:
        dk = json.load(f)
    K    = np.array(dk["matriz_K"], dtype=np.float64)
    dist = np.array(dk.get("distorcao", DISTORCAO), dtype=np.float64)
    nota = dk.get("metodo", f"JSON: {os.path.basename(p)}")
    return K, dist, nota

def montar_K(largura: int, altura: int) -> Tuple[np.ndarray, np.ndarray, str]:
    dist = np.array(DISTORCAO, dtype=np.float64)
    if METODO_K == "json":      return K_de_json(largura, altura)
    elif METODO_K == "fisico":  K, nota = K_por_parametros_fisicos(largura, altura)
    elif METODO_K == "fov":     K, nota = K_por_fov(largura, altura)
    elif METODO_K == "heuristica": K, nota = K_heuristica(largura, altura)
    else: sys.exit(f"[ERRO] METODO_K desconhecido: '{METODO_K}'")
    return K, dist, nota

def refinar_K_com_pares(pares, K, dist, largura, altura):
    if len(pares) < 6:
        return None, None, float("inf"), "Mínimo 6 pares necessários."
    pts3d = np.array([p.ponto_3d for p in pares], dtype=np.float64)
    pts2d = np.array([p.ponto_2d for p in pares], dtype=np.float64)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
    try:
        erro, K_new, dist_new, _, _ = cv2.calibrateCamera(
            [pts3d.reshape(-1,1,3)], [pts2d.reshape(-1,1,2)],
            (largura, altura), K.copy(), dist.copy(), flags=flags)
    except cv2.error as e:
        return None, None, float("inf"), f"Falha: {e}"
    return K_new, dist_new.ravel(), erro, f"K refinada via {len(pares)} pares. Erro={erro:.3f}px"

def salvar_K_json(K, dist, largura, altura, nota, caminho):
    dados = {
        "resolucao": {"largura": largura, "altura": altura},
        "metodo": nota, "matriz_K": K.tolist(),
        "fx": K[0,0], "fy": K[1,1], "cx": K[0,2], "cy": K[1,2],
        "distorcao": list(dist),
    }
    with open(caminho, "w") as f:
        json.dump(dados, f, indent=4)
    print(f"  [OK] Matriz K salva em: {caminho}")
# Preview 3D
class Preview3D:
    def __init__(self, mesh_orig, largura=900, altura=650):
        self._mesh_orig = mesh_orig
        self._largura   = largura
        self._altura    = altura
        self._lock      = threading.Lock()
        self._rvec      = None
        self._pose_nova = False
        self._parar     = False
        self._thread    = None

    def iniciar(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def atualizar_pose(self, rvec):
        with self._lock:
            self._rvec      = rvec.ravel().copy()
            self._pose_nova = True

    def parar(self):
        self._parar = True
        if self._thread: self._thread.join(timeout=3.0)

    def _R_para_open3d(self, rvec):
        R_cv, _ = cv2.Rodrigues(rvec)
        R_mundo_cv = R_cv.T
        return _R_CV2O3D @ R_mundo_cv @ _R_CV2O3D.T

    def _loop(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Preview 3D — rotação em tempo real",
                          width=self._largura, height=self._altura)
        opt = vis.get_render_option()
        opt.background_color    = np.array([0.10, 0.10, 0.13])
        opt.mesh_show_back_face = True
        opt.light_on            = True
        eixos = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
        vis.add_geometry(eixos)
        mesh = o3d.geometry.TriangleMesh(self._mesh_orig)
        mesh.compute_vertex_normals()
        vis.add_geometry(mesh)
        vis.reset_view_point(True)
        R_atual = np.eye(3)
        while not self._parar:
            with self._lock:
                tem_nova = self._pose_nova
                if tem_nova:
                    rvec = self._rvec.copy(); self._pose_nova = False
            if tem_nova:
                R_novo = self._R_para_open3d(rvec)
                centro = mesh.get_center()
                mesh.rotate(R_atual.T, center=centro)
                mesh.rotate(R_novo,   center=centro)
                R_atual = R_novo
                vis.update_geometry(mesh)
            vis.poll_events(); vis.update_renderer()
            time.sleep(0.016)
        vis.destroy_window()
# Memória de frame + câmera 3D
def _p(d, nome): return os.path.join(d, nome)
def carregar_memoria(d):
    p = _p(d, "memoria_frames.json")
    return json.load(open(p)) if os.path.exists(p) else {}

def salvar_memoria(d, mem):
    with open(_p(d, "memoria_frames.json"), "w") as f: json.dump(mem, f, indent=2)

def salvar_camera3d(vis, d):
    try:
        ctrl = vis.get_view_control()
        p    = ctrl.convert_to_pinhole_camera_parameters()
        data = {"extrinsic": p.extrinsic.tolist(), "intrinsic": {
            "width": p.intrinsic.width, "height": p.intrinsic.height,
            "intrinsic_matrix": p.intrinsic.intrinsic_matrix.tolist()}}
        with open(_p(d, "camera3d.json"), "w") as f: json.dump(data, f, indent=2)
    except Exception as e: print(f"    [AVISO] Não salvou câmera 3D: {e}")

def restaurar_camera3d(vis, d):
    p = _p(d, "camera3d.json")
    if not os.path.exists(p): return
    try:
        data = json.load(open(p)); ctrl = vis.get_view_control()
        par  = ctrl.convert_to_pinhole_camera_parameters()
        par.extrinsic = np.array(data["extrinsic"])
        idata = data["intrinsic"]
        par.intrinsic.set_intrinsics(
            idata["width"], idata["height"],
            idata["intrinsic_matrix"][0][0], idata["intrinsic_matrix"][1][1],
            idata["intrinsic_matrix"][0][2], idata["intrinsic_matrix"][1][2])
        ctrl.convert_from_pinhole_camera_parameters(par, allow_arbitrary=True)
        print("    [OK] Vista 3D restaurada.")
    except Exception as e: print(f"    [AVISO] Não restaurou câmera 3D: {e}")
# Open3D — seleção de vértice
def _esfera(centro, cor_rgb, raio=0.008):
    e = o3d.geometry.TriangleMesh.create_sphere(radius=raio)
    e.translate(centro); e.paint_uniform_color(list(cor_rgb))
    e.compute_vertex_normals(); return e

def selecionar_um_ponto_3d(mesh, label, pares_existentes, script_dir):
    # Opção 1: subdivisão para mais vértices selecionáveis
    if SUBDIVIDIR_MALHA:
        mesh_sel = o3d.geometry.TriangleMesh(mesh)
        if SUBDIVISAO_METODO == "loop":
            mesh_sel = mesh_sel.subdivide_loop(number_of_iterations=SUBDIVISAO_ITERACOES)
        else:
            mesh_sel = mesh_sel.subdivide_midpoint(number_of_iterations=SUBDIVISAO_ITERACOES)
        mesh_sel.compute_vertex_normals()
        n_orig = len(np.asarray(mesh.vertices))
        n_novo = len(np.asarray(mesh_sel.vertices))
        print(f"    [SUBDIV] {n_orig} → {n_novo} vértices ({SUBDIVISAO_METODO} ×{SUBDIVISAO_ITERACOES})")
    else:
        mesh_sel = mesh
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(
        window_name=f"Ponto {label} — SHIFT+Clique no vértice | feche para confirmar",
        width=1100, height=750)
    vis.add_geometry(mesh_sel)
    for par in pares_existentes:
        vis.add_geometry(_esfera(par.ponto_3d, CORES_RGB[(par.label-1) % len(CORES_RGB)]))
    vis.poll_events(); vis.update_renderer()
    restaurar_camera3d(vis, script_dir)
    vis.poll_events(); vis.update_renderer()
    print(f"    [3D] SHIFT+Clique no vértice do Ponto {label} e feche a janela.")
    vis.run()
    salvar_camera3d(vis, script_dir)
    vis.destroy_window()
    picked = vis.get_picked_points()
    if not picked: print("    [AVISO] Nenhum vértice selecionado."); return None
    coord = list(picked[0].coord)
    print(f"    [OK] Ponto {label} 3D: {[f'{v:.4f}' for v in coord]}")
    return coord
# Redimensionamento
def escala_para_tela(frame, max_w=1280, max_h=700):
    h, w = frame.shape[:2]
    escala = min(max_w / w, max_h / h, 1.0)
    if escala < 1.0:
        frame = cv2.resize(frame, (int(w*escala), int(h*escala)),
                           interpolation=cv2.INTER_AREA)
    return frame, escala

def tela_para_orig(pt, escala): return [pt[0] / escala, pt[1] / escala]
def orig_para_tela(pt, escala): return (int(pt[0] * escala), int(pt[1] * escala))
# Tracker Lucas-Kanade
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def atualizar_tracker(pares, gray_prev, gray):
    if not pares or gray_prev is None: return pares, False
    pts = np.array([p.ponto_2d for p in pares], dtype=np.float32).reshape(-1, 1, 2)
    novos, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray, pts, None, **LK)
    if novos is None: return pares, False
    pares_ok = [Par(p.ponto_3d, pt.tolist(), p.label)
                for p, st, pt in zip(pares, status.ravel(), novos.reshape(-1, 2)) if st]
    perdidos = len(pares) - len(pares_ok)
    if perdidos: print(f"    [TRACKER] {perdidos} ponto(s) perdido(s).")
    return pares_ok, len(pares_ok) >= 4
# solvePnP
def calcular_pose(pares, K, dist):
    if len(pares) < 4: return None
    pts3d = np.array([p.ponto_3d for p in pares], dtype=np.float64)
    pts2d = np.array([p.ponto_2d for p in pares], dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, dist,
        iterationsCount=300, reprojectionError=8.0, confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    if inliers is not None and len(inliers) >= 4:
        idx = inliers.ravel()
        cv2.solvePnP(pts3d[idx], pts2d[idx], K, dist, rvec, tvec,
                     useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    return rvec, tvec

def fazer_registro(rvec, tvec, ts, frame_n, n_pts):
    rot = Rotation.from_matrix(cv2.Rodrigues(rvec)[0])
    return RegistroPose(
        timestamp_s=round(ts, 4), numero_frame=frame_n,
        rvec=rvec.ravel().tolist(), tvec=tvec.ravel().tolist(),
        quaternion=rot.as_quat().tolist(),
        euler_graus=rot.as_euler("xyz", degrees=True).tolist(),
        sucesso=True, num_pontos=n_pts)

def erro_reproj_medio(pares, K, dist, rvec, tvec) -> float:
    pts3d = np.array([p.ponto_3d for p in pares], dtype=np.float64)
    pts2d = np.array([p.ponto_2d for p in pares], dtype=np.float64)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    return float(np.mean(np.linalg.norm(pts2d - proj.reshape(-1, 2), axis=1)))
# Desenho / HUD
def cor_par(label): return CORES_BGR[(label-1) % len(CORES_BGR)]

def desenhar_eixos(frame, K, dist, rvec, tvec, escala, comp=0.1):
    pts = np.float32([[0,0,0],[comp,0,0],[0,comp,0],[0,0,comp]])
    proj, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    proj = (proj.reshape(-1, 2) * escala).astype(int)
    o = tuple(proj[0])
    cv2.arrowedLine(frame, o, tuple(proj[1]), (0,   0, 255), 2, tipLength=0.2)
    cv2.arrowedLine(frame, o, tuple(proj[2]), (0, 255,   0), 2, tipLength=0.2)
    cv2.arrowedLine(frame, o, tuple(proj[3]), (255,  0,   0), 2, tipLength=0.2)

def desenhar_pares(frame, pares, escala):
    for par in pares:
        pt = orig_para_tela(par.ponto_2d, escala)
        c  = cor_par(par.label)
        cv2.circle(frame, pt, 7, c, -1)
        cv2.circle(frame, pt, 8, (255, 255, 255), 1)
        cv2.putText(frame, str(par.label), (pt[0]+10, pt[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, str(par.label), (pt[0]+10, pt[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)

def hud(frame, pares, frame_n, total, ts, n_reg, pausado,
        instrucao="", erro_reproj=None, nota_K=""):
    status  = "[PAUSADO]" if pausado else "[TRACKING]"
    err_txt = f"  Reproj={erro_reproj:.1f}px" if erro_reproj is not None else ""
    linhas  = [
        f"Frame {frame_n}/{total}  t={ts:.2f}s  Pares={len(pares)}  "
        f"Registros={n_reg}  {status}{err_txt}",
        "A=add(imã)  D=del  R=reset  K=refinarK  SETAS=navegar  ESPACO=pausar  Q=sair",
    ]
    if nota_K:   linhas.append(f"K: {nota_K[:90]}")
    if instrucao: linhas.append(instrucao)
    for i, txt in enumerate(linhas):
        y   = 24 + i * 24
        cor = (0, 220, 255) if (i == len(linhas)-1 and instrucao) else \
              (180, 255, 180) if (i == 2 and nota_K) else (255, 220, 50)
        cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    cor, 1, cv2.LINE_AA)

def hud_frame_inicial(frame, frame_n, total, ts, frame_mem):
    mem_txt = f"  (lembrado: {frame_mem})" if frame_mem is not None else ""
    linhas  = [
        f"SELEÇÃO DO FRAME INICIAL{mem_txt}",
        f"Frame {frame_n}/{total}   t={ts:.2f}s",
        "→/← = ±10  (segurar acelera)   ENTER = confirmar   Q = cancelar",
    ]
    for i, txt in enumerate(linhas):
        y   = 30 + i * 28
        cor = (100, 255, 100) if i == 0 else (255, 220, 50)
        cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    cor, 1, cv2.LINE_AA)
# Fase 1 — seleção de frame inicial
def selecionar_frame_inicial(cap, fps, total_frames, frame_lembrado, JANELA):
    frame_n  = frame_lembrado if frame_lembrado is not None else 0
    KC_DIR = 0x270000; KC_ESQ = 0x250000; KC_ENTER = 13
    ultima_seta = None; repeticoes = 0

    def ler(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, f = cap.read()
        return f if ret else None

    print("\n[FRAME INICIAL] Navegue com as setas e confirme com ENTER.")
    while True:
        frame = ler(frame_n)
        if frame is None: frame_n = 0; continue
        exib, _ = escala_para_tela(frame)
        hud_frame_inicial(exib, frame_n, total_frames, frame_n / fps, frame_lembrado)
        cv2.imshow(JANELA, exib)
        key_ex = cv2.waitKeyEx(80)
        if key_ex == -1: ultima_seta = None; repeticoes = 0; continue
        if key_ex == KC_ENTER or (key_ex & 0xFF) in (13, 10):
            print(f"  [OK] Frame confirmado: {frame_n}"); return frame_n
        if (key_ex & 0xFF) == ord('q'): return None
        if key_ex in (KC_DIR, KC_ESQ):
            sinal = +1 if key_ex == KC_DIR else -1
            repeticoes = (repeticoes + 1) if key_ex == ultima_seta else 0
            ultima_seta = key_ex
            delta = min(10 + repeticoes * 10, 100)
            frame_n = max(0, min(total_frames - 1, frame_n + sinal * delta))
        else: ultima_seta = None; repeticoes = 0
# Loop principal
def ponto_o3d_para_opencv(pt: list) -> list:
    """Converte coordenada de vértice Open3D para o espaço esperado pelo solvePnP."""
    x, y, z = pt
    return [x, -y, -z]

def loop_video(caminho_video, K, dist, nota_K, caminho_saida_K,
               caminho_obj, caminho_saida, script_dir):

    print(f"[OK] Carregando modelo 3D: {caminho_obj}")
    mesh = o3d.io.read_triangle_mesh(caminho_obj)
    mesh.compute_vertex_normals()

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened(): sys.exit(f"[ERRO] Não foi possível abrir: {caminho_video}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    largura_vid  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_vid   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    JANELA = "Estimador de Pose"
    cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)

    memoria        = carregar_memoria(script_dir)
    chave_video    = os.path.basename(caminho_video)
    frame_lembrado = memoria.get(chave_video)
    frame_inicial  = selecionar_frame_inicial(cap, fps, total_frames, frame_lembrado, JANELA)
    if frame_inicial is None: cap.release(); cv2.destroyAllWindows(); return

    memoria[chave_video] = frame_inicial
    salvar_memoria(script_dir, memoria)

    preview = None
    if USAR_PREVIEW_3D:
        preview = Preview3D(mesh)
        preview.iniciar()
        print("  [PREVIEW 3D] Janela aberta — exibindo apenas rotação.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicial)
    ret, frame_orig = cap.read()
    if not ret: sys.exit("[ERRO] Não foi possível ler o frame inicial.")

    pares:          List[Par] = []
    registros:      List[RegistroPose] = []
    gray_prev       = None
    pausado         = True
    modo            = "SELECAO"
    proximo_label   = 1
    ult_rvec        = None
    ult_tvec        = None
    frame_atual     = frame_inicial
    instrucao_hud   = "Pressione [A] para adicionar o primeiro par de pontos"
    ult_erro_reproj = None
# Estado do mouse com imã
    clique = EstadoCliqueIman()
    cv2.setMouseCallback(JANELA, clique.callback)

    # Features detectados no frame atual (coord. de tela)
    features_tela_cache: Optional[np.ndarray] = None
    modo_iman = False   # True apenas durante o modo [A]

    def ler_frame_n(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        r, f = cap.read()
        return f if r else None

    print(f"\n[INICIO] K ativa: {nota_K}")
    print("         [A] adicionar pares com imã (mín. 4) → [ESPAÇO] iniciar tracking\n")

    while True:
        exib, escala = escala_para_tela(frame_orig)

        # Pose
        if len(pares) >= 4:
            res = calcular_pose(pares, K, dist)
            if res:
                ult_rvec, ult_tvec = res
                ts  = frame_atual / fps
                reg = fazer_registro(ult_rvec, ult_tvec, ts, frame_atual, len(pares))
                registros[:] = [r for r in registros if r.numero_frame != frame_atual]
                registros.append(reg)
                registros.sort(key=lambda r: r.numero_frame)
                ult_erro_reproj = erro_reproj_medio(pares, K, dist, ult_rvec, ult_tvec)
                if preview: preview.atualizar_pose(ult_rvec)

        if ult_rvec is not None:
            comp = float(np.linalg.norm(ult_tvec)) * 0.15 or 0.05
            desenhar_eixos(exib, K, dist, ult_rvec, ult_tvec, escala, comp)

        desenhar_pares(exib, pares, escala)

        # Overlay do imã (features + cursor) — só no modo [A]
        if modo_iman:
            desenhar_features(exib, features_tela_cache)
            desenhar_cursor_iman(exib, clique.pt_mouse, features_tela_cache)

        hud(exib, pares, frame_atual, total_frames, frame_atual/fps,
            len(registros), pausado, instrucao_hud, ult_erro_reproj, nota_K)
        cv2.imshow(JANELA, exib)

        espera = 1 if (not pausado and modo == "TRACKING") else 40
        key_ex = cv2.waitKeyEx(espera)
        key    = key_ex & 0xFF

        # Navegação pausada
        if pausado and not modo_iman:
            delta = None
            if   key_ex == 0x270000: delta = +10
            elif key_ex == 0x250000: delta = -10
            if delta is not None:
                novo = max(0, min(total_frames-1, frame_atual + delta))
                f = ler_frame_n(novo)
                if f is not None:
                    frame_orig  = f
                    frame_atual = novo
                    gray_prev   = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
                    instrucao_hud = f"Frame {frame_atual} — ESPACO para iniciar tracking"
                continue

        if key == ord('q'): break
# ESPAÇO
        elif key == ord(' '):
            if pausado:
                if len(pares) < 4:
                    instrucao_hud = f"Adicione mais {4-len(pares)} par(es) com [A]"
                else:
                    pausado = False; modo = "TRACKING"
                    gray_prev = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
                    instrucao_hud = ""
                    print(f"  [TRACKING] Iniciando frame {frame_atual}...")
            else:
                pausado = True; modo = "SELECAO"
                instrucao_hud = "Pausado — setas=navegar  A=add  D=del  R=reset  K=refinarK"
# K — refinar K
        elif key == ord('k'):
            if len(pares) < 6:
                instrucao_hud = f"Refinar K requer ≥ 6 pares. Atual: {len(pares)}."
            elif ult_rvec is None:
                instrucao_hud = "Precisa de uma pose calculada antes de refinar K."
            else:
                print(f"\n  [K] Refinando K com {len(pares)} pares...")
                K_new, dist_new, erro_cal, nota_new = refinar_K_com_pares(
                    pares, K, dist, largura_vid, altura_vid)
                if K_new is not None:
                    K = K_new; dist = dist_new; nota_K = nota_new
                    salvar_K_json(K, dist, largura_vid, altura_vid, nota_K, caminho_saida_K)
                    instrucao_hud = (f"K refinada! Erro={erro_cal:.2f}px — "
                                     f"salva em {os.path.basename(caminho_saida_K)}")
                    print(f"  [OK] {nota_new}")
                else:
                    instrucao_hud = f"Refinamento falhou: {nota_new}"
# A — adicionar par COM IMÃ
        elif key == ord('a'):
            label         = proximo_label
            pausado_antes = pausado
            pausado       = True
            modo_iman     = True

            # Detecta features no frame atual (coord. originais → escala p/ tela)
            gray_frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
            feats_orig = detectar_features(gray_frame)
            if feats_orig is not None:
                features_tela_cache = feats_orig * escala  # escala p/ coordenadas de tela
            else:
                features_tela_cache = None

            clique.set_features(features_tela_cache)
            clique.reset()

            n_feats = len(features_tela_cache) if features_tela_cache is not None else 0
            instrucao_hud = (f"Ponto {label}: clique na imagem  "
                             f"[imã ativo — {n_feats} features]  ESC=cancelar")
            print(f"\n  [A] Ponto {label} — imã ativo ({n_feats} features). Clique no vídeo...")

            # Aguarda clique com loop de refresh (para animação do cursor)
            while not clique.pronto:
                exib2, _ = escala_para_tela(frame_orig)
                desenhar_pares(exib2, pares, escala)
                desenhar_features(exib2, features_tela_cache)
                desenhar_cursor_iman(exib2, clique.pt_mouse, features_tela_cache)
                hud(exib2, pares, frame_atual, total_frames, frame_atual/fps,
                    len(registros), True, instrucao_hud, nota_K=nota_K)
                cv2.imshow(JANELA, exib2)
                k2 = cv2.waitKey(16) & 0xFF
                if k2 == 27:  # ESC cancela
                    break

            modo_iman = False
            clique.set_features(None)

            if not clique.pronto:
                instrucao_hud = "Adição cancelada."; pausado = pausado_antes; continue

            # Ponto em coordenada de tela (já snappado)
            pt_tela = clique.pt_clique
            pt_orig = tela_para_orig(pt_tela, escala)
            clique.reset()

            # Mostra confirmação visual
            exib3, _ = escala_para_tela(frame_orig)
            desenhar_pares(exib3, pares, escala)
            cv2.circle(exib3, pt_tela, 11, (0, 255, 255), 2)
            cv2.circle(exib3, pt_tela,  7, cor_par(label), -1)
            cv2.circle(exib3, pt_tela,  8, (255, 255, 255), 1)
            cv2.putText(exib3, str(label), (pt_tela[0]+11, pt_tela[1]-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            hud(exib3, pares, frame_atual, total_frames, frame_atual/fps,
                len(registros), True,
                f"Ponto {label} marcado — selecione vértice no modelo 3D...",
                nota_K=nota_K)
            cv2.imshow(JANELA, exib3); cv2.waitKey(200)

            pt3d_o3d = selecionar_um_ponto_3d(mesh, label, pares, script_dir)
            if pt3d_o3d is None:
                instrucao_hud = f"Ponto {label} cancelado."; pausado = pausado_antes; continue

            pt3d = ponto_o3d_para_opencv(pt3d_o3d)  # ← conversão aqui

            pares.append(Par(ponto_3d=pt3d, ponto_2d=pt_orig, label=label))
            proximo_label += 1
            gray_prev = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)

            n_falta = max(0, 4 - len(pares))
            instrucao_hud = (f"Par {label} adicionado! Total: {len(pares)}. " +
                             (f"Faltam {n_falta}." if n_falta else
                              ("Pressione [K] para refinar K." if len(pares) >= 6
                               else "Pronto! ESPACO para iniciar.")))
            print(f"  [OK] Par {label}. Total: {len(pares)}.")
            pausado = pausado_antes
# D — deletar par
        elif key == ord('d'):
            if not pares: instrucao_hud = "Nenhum par para deletar."; continue
            pausado_antes = pausado; pausado = True
            instrucao_hud = "DELETAR: clique sobre o ponto"
            exib_d, escala_d = escala_para_tela(frame_orig)
            desenhar_pares(exib_d, pares, escala_d)
            hud(exib_d, pares, frame_atual, total_frames, frame_atual/fps,
                len(registros), True, instrucao_hud, nota_K=nota_K)
            cv2.imshow(JANELA, exib_d)
            clique.reset()
            while not clique.pronto:
                if cv2.waitKey(30) & 0xFF == 27: break
            if clique.pronto:
                mx, my = clique.pt_clique
                dsts = [np.hypot(*(np.array(orig_para_tela(p.ponto_2d, escala_d))-[mx, my]))
                        for p in pares]
                idx = int(np.argmin(dsts))
                if dsts[idx] <= 40:
                    rem = pares.pop(idx)
                    instrucao_hud = f"Par {rem.label} removido."
                    print(f"  [OK] Par {rem.label} removido.")
                else:
                    instrucao_hud = "Nenhum ponto próximo ao clique."
            else:
                instrucao_hud = "Deleção cancelada."
            clique.reset(); pausado = pausado_antes
# R — reset
        elif key == ord('r'):
            pares.clear(); gray_prev = None
            ult_rvec = ult_tvec = None; ult_erro_reproj = None
            pausado = True; modo = "SELECAO"; proximo_label = 1
            instrucao_hud = "Pares resetados. Pressione [A] para recomeçar."
            print(f"  [R] Todos os pares removidos no frame {frame_atual}.")
# Avançar tracking
        if not pausado and modo == "TRACKING":
            ret, prox = cap.read()
            if not ret: print("[FIM] Vídeo concluído."); break
            frame_atual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            gray_novo   = cv2.cvtColor(prox, cv2.COLOR_BGR2GRAY)
            pares, ok   = atualizar_tracker(pares, gray_prev, gray_novo)
            if not ok:
                print(f"  [AVISO] Tracking instável no frame {frame_atual}. Pausando.")
                instrucao_hud = "Tracking perdido — corrija com [A]/[D] e retome"
                pausado = True; modo = "SELECAO"
            gray_prev  = gray_novo
            frame_orig = prox
# Salvar
    cap.release(); cv2.destroyAllWindows()
    if preview: preview.parar()
    if registros:
        registros.sort(key=lambda r: r.numero_frame)
        with open(caminho_saida, "w") as f:
            json.dump([asdict(r) for r in registros], f, indent=2)
        print(f"\n[OK] {len(registros)} registro(s) salvos em: {caminho_saida}")
    else:
        print("\n[AVISO] Nenhum registro de pose gerado.")
# Seleção de arquivos
_tk_root = None
def _get_root():
    global _tk_root
    if _tk_root is None:
        _tk_root = tk.Tk(); _tk_root.withdraw()
    return _tk_root

def pedir_arquivo(titulo, tipos):
    root = _get_root(); root.lift(); root.focus_force()
    return filedialog.askopenfilename(parent=root, title=titulo, filetypes=tipos)

def encontrar_no_dir(diretorio, extensoes):
    for f in sorted(os.listdir(diretorio)):
        if any(f.lower().endswith(e) for e in extensoes):
            return os.path.join(diretorio, f)
    return None
# Entry point
if __name__ == "__main__":
    print("=== Estimador de Pose — solvePnP ===")
    print(f"    Método K: {METODO_K}")
    if METODO_K == "fisico":
        print(f"    focal={FOCAL_MM}mm  sensor={SENSOR_W_MM}×{SENSOR_H_MM}mm")
    elif METODO_K == "fov":
        print(f"    FOV horizontal={FOV_H_GRAUS}°")
    elif METODO_K == "json":
        print(f"    Arquivo K: {CAMINHO_K_JSON}")
    print(f"    Imã: raio={IMAN_RAIO_PX}px  features={IMAN_MAX_CORNERS}  "
          f"qualidade={IMAN_QUALIDADE}  dist_min={IMAN_DIST_MIN}px\n")

    aqui = os.path.dirname(os.path.abspath(__file__))

    v = pedir_arquivo("Selecione o vídeo",
                      [("Vídeos", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Todos", "*.*")])
    if not v: sys.exit("Cancelado.")
    print(f"[OK] Vídeo: {v}")

    _cap = cv2.VideoCapture(v)
    _w   = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _h   = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _cap.release()

    obj = encontrar_no_dir(aqui, [".obj"])
    if obj: print(f"[AUTO] Modelo encontrado: {obj}")
    else:
        obj = pedir_arquivo("Selecione o modelo .obj",
                            [("OBJ", "*.obj"), ("Todos", "*.*")])
        if not obj: sys.exit("Cancelado.")
    print(f"[OK] Modelo: {obj}")

    K, dist, nota_K = montar_K(_w, _h)
    print(f"[OK] Matriz K ({nota_K})")
    print(f"     fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    caminho_saida_K = os.path.join(aqui, "matriz_K.json")
    salvar_K_json(K, dist, _w, _h, nota_K, caminho_saida_K)

    saida = os.path.join(aqui, "pose_dados.json")
    print(f"[OK] Saída: {saida}\n") 

    loop_video(v, K, dist, nota_K, caminho_saida_K, obj, saida, aqui)