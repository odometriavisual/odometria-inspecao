"""
Script 4: Preview de Pose — Objeto 3D Real + Vídeo Lado a Lado
==============================================================

FLUXO DE USO:
  1. Deixe REMAPEAR_EIXO = False
  2. Rode e observe qual eixo (X/Y/Z) do cubo gira do jeito que você quer
  3. Configure EIXO_FONTE com esse eixo (0=X, 1=Y, 2=Z)
  4. Configure EIXO_DESTINO com onde quer exibir (0=X vermelho, 1=Y verde, 2=Z azul)
  5. Se girar ao contrário, mude ESCALA para -1.0
  6. Ative REMAPEAR_EIXO = True

Dependências: open3d, numpy, scipy, opencv-python
"""

import cv2
import numpy as np
import open3d as o3d
import json
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog
from scipy.spatial.transform import Rotation, Slerp
from typing import List, Optional


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
#  CONFIGURAÇÕES
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

USAR_CUBO     = True
TAMANHO_CUBO  = 0.10
TAMANHO_EIXOS = 0.12

LARGURA_PREVIEW = 900
ALTURA_PREVIEW  = 700
LARGURA_VIDEO   = 600
ALTURA_VIDEO    = 600

# ─────────────────────────────────────────────────────────────────────────────
# REMAPEAMENTO DE EIXO
# ─────────────────────────────────────────────────────────────────────────────

# False = modo diagnóstico: rvec bruto direto, sem nenhuma transformação
# True  = ativa o remapeamento abaixo
REMAPEAR_EIXO = False

# Qual componente do rvec bruto contém a rotação que você quer
# 0 = X  |  1 = Y  |  2 = Z
EIXO_FONTE = 1

# Sentido: 1.0 = mesmo sentido do rvec bruto  |  -1.0 = inverte
ESCALA = -1.0

# Em qual eixo do Open3D aplicar a rotação
# 0 = X (vermelho)  |  1 = Y (verde, aponta pra cima)  |  2 = Z (azul)
EIXO_DESTINO = 1

# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
#  FIM DAS CONFIGURAÇÕES
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# Geometrias
# ─────────────────────────────────────────────────────────────────────────────

ROT_OBJ_X = -90
ROT_OBJ_Y = 0
ROT_OBJ_Z = 0

def carregar_obj(caminho: str) -> o3d.geometry.TriangleMesh:
    # enable_post_processing=True ajuda a carregar materiais do MTL
    mesh = o3d.io.read_triangle_mesh(caminho, enable_post_processing=True)
    
    if not mesh.has_textures():
        print("[AVISO] O Open3D não identificou texturas no MTL.")
    else:
        print("[SUCESSO] Texturas carregadas com sucesso!")

    # Se o modelo aparecer escuro, calculamos as normais
    mesh.compute_vertex_normals()

    # Centralização e Escala (Mantendo sua lógica original)
    centro = mesh.get_center()
    mesh.translate(-centro)
    bb = mesh.get_axis_aligned_bounding_box()
    escala = 1.0 / max(bb.get_extent())
    mesh.scale(escala, center=[0, 0, 0])
    
    # Rotação de eixos do script
    R = Rotation.from_euler('xyz', [ROT_OBJ_X, ROT_OBJ_Y, ROT_OBJ_Z], degrees=True).as_matrix()
    mesh.rotate(R, center=[0, 0, 0])
    
    return mesh

def criar_cubo_orientado(tamanho: float = 0.08) -> o3d.geometry.TriangleMesh:
    """Cubo com faces coloridas por eixo e setas X/Y/Z."""
    s = tamanho / 2.0

    vertices = np.array([
        [-s,-s,-s],[+s,-s,-s],[+s,+s,-s],[-s,+s,-s],
        [-s,-s,+s],[+s,-s,+s],[+s,+s,+s],[-s,+s,+s],
    ], dtype=np.float64)

    triangles = np.array([
        [0,2,1],[0,3,2],
        [4,5,6],[4,6,7],
        [0,1,5],[0,5,4],
        [2,3,7],[2,7,6],
        [0,4,7],[0,7,3],
        [1,2,6],[1,6,5],
    ], dtype=np.int32)

    face_colors = [
        [0.30,0.30,0.80],[0.30,0.30,0.80],
        [0.10,0.10,1.00],[0.10,0.10,1.00],
        [0.30,0.80,0.30],[0.30,0.80,0.30],
        [0.00,0.55,0.00],[0.00,0.55,0.00],
        [0.80,0.30,0.30],[0.80,0.30,0.30],
        [0.90,0.05,0.05],[0.90,0.05,0.05],
    ]

    verts_exp, tris_exp, cols_exp = [], [], []
    for i, tri in enumerate(triangles):
        base = len(verts_exp)
        verts_exp.extend(vertices[tri])
        tris_exp.append([base, base+1, base+2])
        cols_exp.extend([face_colors[i]] * 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices      = o3d.utility.Vector3dVector(np.array(verts_exp))
    mesh.triangles     = o3d.utility.Vector3iVector(np.array(tris_exp))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(cols_exp))
    mesh.compute_vertex_normals()

    comp   = tamanho * 1.4
    raio_c = tamanho * 0.045
    raio_p = tamanho * 0.11
    alt_p  = tamanho * 0.28
    eixos_def = [
        (np.array([1.,0.,0.]), [0.95,0.15,0.15]),
        (np.array([0.,1.,0.]), [0.15,0.90,0.15]),
        (np.array([0.,0.,1.]), [0.15,0.50,1.00]),
    ]
    setas = o3d.geometry.TriangleMesh()
    for direcao, cor in eixos_def:
        cil = o3d.geometry.TriangleMesh.create_cylinder(
            radius=raio_c, height=comp-alt_p, resolution=12, split=1)
        cil.translate([0., 0., (comp-alt_p)/2.])
        cone = o3d.geometry.TriangleMesh.create_cone(
            radius=raio_p, height=alt_p, resolution=12)
        cone.translate([0., 0., comp-alt_p])
        for part in (cil, cone):
            z     = np.array([0., 0., 1.])
            axis  = np.cross(z, direcao)
            sin_a = np.linalg.norm(axis)
            cos_a = float(np.dot(z, direcao))
            if sin_a > 1e-6:
                axis /= sin_a
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    axis * np.arctan2(sin_a, cos_a))
                part.rotate(R, center=[0, 0, 0])
            elif cos_a < 0:
                part.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
                    np.array([1., 0., 0.]) * np.pi), center=[0, 0, 0])
            part.paint_uniform_color(cor)
            part.compute_vertex_normals()
            setas += part

    return mesh + setas


# ─────────────────────────────────────────────────────────────────────────────
# Tk / busca de arquivos
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Carregar e interpolar poses
# ─────────────────────────────────────────────────────────────────────────────

def carregar_poses(caminho: str) -> List[dict]:
    with open(caminho) as f:
        dados = json.load(f)
    dados = [d for d in dados if d.get("sucesso", False)]
    dados.sort(key=lambda d: d["timestamp_s"])
    print(f"[OK] {len(dados)} poses  "
          f"({dados[0]['timestamp_s']:.2f}s – {dados[-1]['timestamp_s']:.2f}s)")
    return dados


def construir_interpoladores(dados: List[dict]):
    tempos = np.array([d["timestamp_s"] for d in dados])
    quats  = np.array([d["quaternion"]  for d in dados])
    slerp  = Slerp(tempos, Rotation.from_quat(quats))
    return slerp, float(tempos[0]), float(tempos[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo de rotação — lógica central
# ─────────────────────────────────────────────────────────────────────────────

def R_no_tempo(t: float, slerp, t_min: float, t_max: float):
    """
    Retorna (R, rvec) para o instante t.

    REMAPEAR_EIXO=False  →  rvec bruto do solvePnP, sem transformação alguma.
                            Use para diagnóstico: observe qual eixo gira certo.

    REMAPEAR_EIXO=True   →  extrai só o ângulo de EIXO_FONTE, aplica em
                            EIXO_DESTINO com fator ESCALA.
    """
    t    = float(np.clip(t, t_min, t_max))
    rvec = slerp(t).as_rotvec()   # [rx, ry, rz] em radianos

    if REMAPEAR_EIXO:
        # 1. Pega o ângulo escalar do eixo que você escolheu
        angulo = float(rvec[EIXO_FONTE]) * ESCALA

        # 2. Monta o vetor de rotação no eixo destino do Open3D
        rotvec_destino = np.zeros(3)
        rotvec_destino[EIXO_DESTINO] = angulo

        # 3. Constrói a matriz de rotação
        R = Rotation.from_rotvec(rotvec_destino).as_matrix()
        return R, rvec

    # Modo diagnóstico: rvec bruto → matriz direta
    R_bruto, _ = cv2.Rodrigues(rvec.ravel())
    return R_bruto, rvec


# ─────────────────────────────────────────────────────────────────────────────
# Atualizar rotação do objeto na cena
# ─────────────────────────────────────────────────────────────────────────────

_ORIGEM = [0.0, 0.0, 0.0]

def _aplicar_rotacao(obj, R_atual: np.ndarray, R_novo: np.ndarray) -> np.ndarray:
    obj.rotate(R_atual.T, center=_ORIGEM)
    obj.rotate(R_novo,    center=_ORIGEM)
    return R_novo


# ─────────────────────────────────────────────────────────────────────────────
# Captura de frame Open3D (Windows-compatível)
# ─────────────────────────────────────────────────────────────────────────────

def _capturar_frame_o3d(vis, largura: int, altura: int,
                        tentativas: int = 5) -> np.ndarray:
    for _ in range(tentativas):
        for __ in range(3):
            vis.poll_events()
            vis.update_renderer()
        buf = vis.capture_screen_float_buffer(do_render=True)
        img = np.asarray(buf)
        if img.size == 0 or img.ndim < 3:
            time.sleep(0.05)
            continue
        bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if bgr.shape[1] != largura or bgr.shape[0] != altura:
            bgr = cv2.resize(bgr, (largura, altura), interpolation=cv2.INTER_AREA)
        return bgr
    print("\n  [AVISO] Captura falhou — usando frame preto.")
    return np.zeros((altura, largura, 3), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Composição de frames
# ─────────────────────────────────────────────────────────────────────────────

def _compor_lado_a_lado(frame_video, frame_obj, altura_alvo):
    def _rh(img, h):
        r = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*r), h),
                          interpolation=cv2.INTER_AREA)
    v   = _rh(frame_video, altura_alvo)
    c   = _rh(frame_obj,   altura_alvo)
    div = np.full((altura_alvo, 3, 3), [60, 60, 60], dtype=np.uint8)
    return np.hstack([v, div, c])


def _compor_overlay(frame_video, frame_obj, tamanho_rel=0.32, posicao="td"):
    h_v, w_v = frame_video.shape[:2]
    tam    = int(min(h_v, w_v) * tamanho_rel)
    obj_s  = cv2.resize(frame_obj, (tam, tam), interpolation=cv2.INTER_AREA)
    m      = int(tam * 0.05)
    coords = {
        "td": (w_v-tam-m, m),
        "te": (m, m),
        "bd": (w_v-tam-m, h_v-tam-m),
        "be": (m, h_v-tam-m),
    }
    x0, y0 = coords.get(posicao, (w_v-tam-m, m))
    saida  = frame_video.copy()
    cv2.rectangle(saida, (x0-2, y0-2), (x0+tam+2, y0+tam+2), (200, 200, 200), 2)
    gray    = cv2.cvtColor(obj_s, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
    alpha   = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.
    alpha   = np.where(alpha > 0.5, 1.0, 0.40)
    roi     = saida[y0:y0+tam, x0:x0+tam].astype(np.float32)
    blend   = roi*(1-alpha) + obj_s.astype(np.float32)*alpha
    saida[y0:y0+tam, x0:x0+tam] = blend.astype(np.uint8)
    return saida


# ─────────────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────────────

def _hud(frame, t: float, i: int, n: int, rvec: np.ndarray):
    nomes = ["X", "Y", "Z"]
    angulo_fonte = float(rvec[EIXO_FONTE]) if REMAPEAR_EIXO else 0.0
    linhas = [
        f"t={t:.2f}s   frame {i+1}/{n}",
        f"rvec[{nomes[EIXO_FONTE]}]={np.degrees(angulo_fonte):.1f}°  "
        f"→  destino {nomes[EIXO_DESTINO]} × {ESCALA}"
        if REMAPEAR_EIXO else
        f"rvec bruto: {np.degrees(rvec[0]):.1f}°  {np.degrees(rvec[1]):.1f}°  {np.degrees(rvec[2]):.1f}°"
    ]
    for j, txt in enumerate(linhas):
        y = 28 + j*24
        cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 220, 50), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Leitura de frame por timestamp
# ─────────────────────────────────────────────────────────────────────────────

def _ler_frame_por_timestamp(cap, t_segundos: float,
                              fps_orig: float) -> Optional[np.ndarray]:
    frame_n = int(round(t_segundos * fps_orig))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
    ret, frame = cap.read()
    return frame if ret else None


# ─────────────────────────────────────────────────────────────────────────────
# Gerar vídeo
# ─────────────────────────────────────────────────────────────────────────────

def renderizar_video(obj_orig: o3d.geometry.TriangleMesh,
                     dados: List[dict],
                     caminho_saida: str,
                     caminho_video_orig: Optional[str] = None,
                     fps: float = 30.0,
                     largura_obj: int = LARGURA_VIDEO,
                     altura_obj:  int = ALTURA_VIDEO,
                     modo_composicao: str = "lado",
                     posicao_overlay: str = "td"):

    slerp, t_min, t_max = construir_interpoladores(dados)
    n_frames = int((t_max - t_min) * fps) + 1

    nomes_eixo = {0: "X (vermelho)", 1: "Y (verde)", 2: "Z (azul)"}
    print(f"\n[VÍDEO] {n_frames} frames @ {fps:.0f} fps")
    print(f"  Intervalo: {t_min:.2f}s – {t_max:.2f}s  (duração: {t_max-t_min:.1f}s)")
    if REMAPEAR_EIXO:
        print(f"  fonte={EIXO_FONTE}  destino={nomes_eixo[EIXO_DESTINO]}  escala={ESCALA}")
    else:
        print("  [DIAGNÓSTICO] rvec bruto")

    cap = None
    fps_orig = fps
    if caminho_video_orig and os.path.exists(caminho_video_orig):
        cap      = cv2.VideoCapture(caminho_video_orig)
        fps_orig = cap.get(cv2.CAP_PROP_FPS) or fps
        print(f"  Vídeo: {os.path.basename(caminho_video_orig)}  FPS={fps_orig:.3f}")
    elif caminho_video_orig:
        print("  [AVISO] Vídeo não encontrado — gerando só o objeto.")

    # IMPORTANTE: Usamos o objeto original diretamente. 
    # A cópia TriangleMesh(obj_orig) pode descartar o buffer de texturas em algumas versões.
    obj   = obj_orig 
    eixos = o3d.geometry.TriangleMesh.create_coordinate_frame(size=TAMANHO_EIXOS)

    # 1. Inicializa o Visualizador
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Renderizando… (não feche esta janela)",
                      width=largura_obj, height=altura_obj, visible=True)
    
    # 2. Adiciona Geometria
    vis.add_geometry(obj)

    vis.get_render_option().background_color = np.array([0.13, 0.13, 0.13])
    vis.get_render_option().mesh_show_back_face = True

    # ─── AJUSTE DE INCLINAÇÃO DA VISTA (20 GRAUS) ──────────────────────────
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0.0, 0.4, 1.0]) 
    
    view_ctl.set_lookat([0.0, 0.0, 0.0]) # Foca no centro do objeto
    view_ctl.set_up([0.0, 1.0, 0.0])    # Mantém o eixo Y como "cima"
    view_ctl.set_zoom(0.7)              # Aproxima um pouco mais (menor = mais perto)
    
    # CRUCIAL: Forçar o renderizador a aceitar a nova câmara antes de começar
    for _ in range(20):
        vis.poll_events()
        vis.update_renderer()
    # ───────────────────────────────────────────────────────────────────────

    for _ in range(15):
        vis.poll_events()
        vis.update_renderer()

    vis.add_geometry(eixos)

    # 3. Configurações de Renderização (Aqui é onde a textura é ativada)
    opt = vis.get_render_option()
    opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
    opt.light_on = True 
    opt.background_color = np.array([0.13, 0.13, 0.13])
    opt.mesh_show_back_face = True

    # Pequeno loop para garantir que o driver de vídeo carregue os Shaders e Materiais
    for _ in range(15):
        vis.poll_events()
        vis.update_renderer()
    time.sleep(0.5)

    R_atual  = np.eye(3)
    R_ini, _ = R_no_tempo(t_min, slerp, t_min, t_max)
    R_atual  = _aplicar_rotacao(obj, R_atual, R_ini)
    
    # Atualiza geometria após a primeira rotação
    vis.update_geometry(obj)
    vis.poll_events()
    vis.update_renderer()

    fc_test  = _capturar_frame_o3d(vis, largura_obj, altura_obj)
    w_out, h_out = largura_obj, altura_obj

    if cap is not None:
        fv_test = _ler_frame_por_timestamp(cap, t_min, fps_orig)
        if fv_test is not None:
            comp_test = (_compor_lado_a_lado(fv_test, fc_test, fv_test.shape[0])
                         if modo_composicao == "lado"
                         else _compor_overlay(fv_test, fc_test, posicao=posicao_overlay))
            w_out, h_out = comp_test.shape[1], comp_test.shape[0]

    print(f"  Resolução de saída: {w_out}×{h_out}")
    print("  Não feche a janela Open3D!\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(caminho_saida, fourcc, fps, (w_out, h_out))

    for i in range (10):
        vis.update_geometry(obj)

        view_ctl = vis.get_view_control()
        view_ctl.set_front([0.0, 0.4, 1.0]) # A inclinação que você quer
        view_ctl.set_lookat([0.0, 0.0, 0.0])
        view_ctl.set_up([0.0, 1.0, 0.0])
        
        vis.poll_events()
        vis.update_renderer()

    for i in range(n_frames):
        t = t_min + i / fps

        R_novo, rvec = R_no_tempo(t, slerp, t_min, t_max)
        R_atual      = _aplicar_rotacao(obj, R_atual, R_novo)

        # Atualiza o objeto no mundo 3D
        vis.update_geometry(obj)

        view_ctl = vis.get_view_control()
        view_ctl.set_front([0.0, 0.4, 1.0]) # A inclinação que você quer
        view_ctl.set_lookat([0.0, 0.0, 0.0])
        view_ctl.set_up([0.0, 1.0, 0.0])

        vis.poll_events()
        vis.update_renderer()
        
        frame_obj = _capturar_frame_o3d(vis, largura_obj, altura_obj)

        frame_video = None
        if cap is not None:
            frame_video = _ler_frame_por_timestamp(cap, t, fps_orig)

        if frame_video is not None:
            saida = (_compor_lado_a_lado(frame_video, frame_obj, frame_video.shape[0])
                     if modo_composicao == "lado"
                     else _compor_overlay(frame_video, frame_obj, posicao=posicao_overlay))
        else:
            saida = np.zeros((h_out, w_out, 3), dtype=np.uint8)
            fc_r  = cv2.resize(frame_obj,
                               (min(w_out, frame_obj.shape[1]),
                                min(h_out, frame_obj.shape[0])))
            x0 = (w_out - fc_r.shape[1]) // 2
            y0 = (h_out - fc_r.shape[0]) // 2
            saida[y0:y0+fc_r.shape[0], x0:x0+fc_r.shape[1]] = fc_r

        _hud(saida, t, i, n_frames, rvec)
        writer.write(saida)

        if (i+1) % max(1, n_frames//40) == 0:
            pct = (i+1) / n_frames * 100
            bar = "█"*int(pct/5) + "░"*(20-int(pct/5))
            print(f"  [{bar}] {pct:.0f}%  frame {i+1}/{n_frames}", end="\r")

    writer.release()
    vis.destroy_window()
    if cap:
        cap.release()
    print(f"\n[OK] Vídeo salvo: {caminho_saida}")

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    nomes_eixo = {0: "X (vermelho)", 1: "Y (verde)", 2: "Z (azul)"}
    print("=== Preview de Pose 3D ===")
    print(f"    Objeto : {'Cubo colorido' if USAR_CUBO else 'Objeto .obj'}")
    if REMAPEAR_EIXO:
        print(f"    fonte  = eixo {EIXO_FONTE} do rvec bruto")
        print(f"    destino= {nomes_eixo[EIXO_DESTINO]}")
        print(f"    escala = {ESCALA}")
    else:
        print("    Modo   : DIAGNÓSTICO (rvec bruto, sem transformação)")
    print()

    aqui = os.path.dirname(os.path.abspath(__file__))

    # ── pose_dados.json ───────────────────────────────────────────────────────
    json_padrao  = os.path.join(aqui, "pose_dados.json")
    caminho_json = json_padrao if os.path.exists(json_padrao) else \
        pedir_arquivo("Selecione pose_dados.json", [("JSON", "*.json")])
    if not caminho_json:
        sys.exit("Cancelado.")
    print(f"[OK] Poses: {caminho_json}")

    dados = carregar_poses(caminho_json)
    if len(dados) < 2:
        sys.exit("[ERRO] Mínimo 2 poses necessárias.")

    # ── Objeto 3D ─────────────────────────────────────────────────────────────
    if USAR_CUBO:
        print("[OK] Usando cubo colorido.")
        obj_orig = criar_cubo_orientado(tamanho=TAMANHO_CUBO)
    else:
        caminho_obj = encontrar_no_dir(aqui, [".obj"])
        if caminho_obj:
            print(f"[AUTO] Modelo encontrado: {caminho_obj}")
        else:
            caminho_obj = pedir_arquivo(
                "Selecione o modelo .obj",
                [("OBJ", "*.obj"), ("Todos", "*.*")])
        if not caminho_obj:
            sys.exit("Cancelado.")
        obj_orig = carregar_obj(caminho_obj)

    # ── Vídeo original (opcional) ─────────────────────────────────────────────
    caminho_video = "ELMLL25-228-OS006000765640-2025-12-15-T-09-40-00-D014.mp4"
    # caminho_video = "CORTES-CENPES\\PONTO 1\\Rotacao\\ELMLL25-228-OS006000765640-2025-12-15-T-19-12-51-D022.mp4"
    fps     = 30

    modo_comp, pos_overlay = "lado", "td"
    print("\nModo de composição:  [1] Lado a lado   [2] Overlay no canto")
    if input("Opção [1/2]: ").strip() == "2":
        modo_comp = "overlay"
        print("Posição: td=topo-direita  te=topo-esquerda  "
                "bd=baixo-direita  be=baixo-esquerda")
        pos_overlay = input("Posição [td]: ").strip().lower() or "td"


    saida_padrao = os.path.join(aqui, "preview_pose.mp4")
    caminho_mp4  = saida_padrao
    print("\n[INICIANDO RENDERIZAÇÃO...]")
    renderizar_video(obj_orig, dados, caminho_mp4,
                        caminho_video_orig=caminho_video,
                        fps=fps,
                        largura_obj=LARGURA_VIDEO,
                        altura_obj=ALTURA_VIDEO,
                        modo_composicao=modo_comp,
                        posicao_overlay=pos_overlay)
    print(f"[CONCLUÍDO] {caminho_mp4}")