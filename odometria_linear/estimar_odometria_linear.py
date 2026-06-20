import cv2
import numpy as np
import json
import os
import re
from tkinter import Tk, filedialog, messagebox

# ================= CONFIGURAÇÕES =================
FAST_FORWARD = 5
SAFE_MARGIN = 20
SESSION_FILE = ".session.json"
WINDOW_NAME = "Odometria OF: [M] Ajuste | [A/D] +/-10 | [Shift+D] +100 | [ESC] Sair"

# ------------------------------------------------------------------
# CONFIGURAÇÕES DE MODO DE OPERAÇÃO
# ------------------------------------------------------------------
# MODO_LABORATORIO = True:
#   - Câmera fixa, sem zoom. Não é necessário selecionar pontos de
#     referência no cano/fundo. O deslocamento do carrinho é usado
#     diretamente, sem subtração de movimento de câmera.
#   - USAR_COMPENSACAO_ZOOM é ignorado (zoom sempre desativado).
#
# MODO_LABORATORIO = False:
#   - Modo campo normal. Requer pontos de referência.
#   - USAR_COMPENSACAO_ZOOM controla a metodologia.
# ------------------------------------------------------------------
MODO_LABORATORIO = False

# Só relevante quando MODO_LABORATORIO = False.
# True  → compensa zoom dinamicamente (bom para vídeos com zoom suave)
# False → assume zoom = 1.0x sempre (sem negação de movimento)
USAR_COMPENSACAO_ZOOM = False
# ------------------------------------------------------------------

# Parâmetros do Lucas-Kanade
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

# Parâmetros de compensação de zoom
ZOOM_DETECTION_MIN_POINTS = 4
ZOOM_SMOOTH_ALPHA = 0.3
ZOOM_THRESHOLD = 0.005

# METODOLOGIA ESCOLHIDA (definida automaticamente abaixo)
METODOLOGIA = None  # "com_zoom", "sem_zoom" ou "laboratorio"


# ================= FUNÇÕES DE APOIO =================

def get_screen_scale(h_orig):
    try:
        root = Tk()
        screen_h = root.winfo_screenheight()
        root.destroy()
        return (screen_h * 0.85) / h_orig
    except:
        return 0.6


def extract_file_info(path):
    name = os.path.basename(path)
    pattern = r"(\d{4}-\d{2}-\d{2})-T-(\d{2})-(\d{2})-(\d{2})"
    match = re.search(pattern, name)
    return (match.group(1), f"{match.group(2)}:{match.group(3)}:{match.group(4)}") if match else (
        "Desconhecida", "Desconhecido")


def load_session(video_name):
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                return json.load(f).get(video_name, 0)
        except:
            return 0
    return 0


def save_session(video_name, frame_idx):
    sessions = {}
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                sessions = json.load(f)
        except:
            pass
    sessions[video_name] = frame_idx
    with open(SESSION_FILE, "w") as f:
        json.dump(sessions, f)


def calculate_diameter(line_sup, line_inf):
    p1, p2 = np.array(line_sup[0]), np.array(line_sup[1])
    p3 = np.array([(line_inf[0][0] + line_inf[1][0]) / 2, (line_inf[0][1] + line_inf[1][1]) / 2])
    d = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    return round(float(d), 2)


def select_line(img, prompt, scale):
    pts = []
    drawing = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal pts, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            pts = [(x, y), (x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            pts[1] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            pts[1] = (x, y)

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    while True:
        display = img.copy()
        cv2.rectangle(display, (0, 0), (display.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(display, prompt, (15, 35), 1, 1.5, (0, 255, 255), 2)
        if len(pts) == 2:
            cv2.line(display, pts[0], pts[1], (0, 0, 255), 2)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        elif key == 27:
            cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)
            return None
    cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)
    return [(int(p[0] / scale), int(p[1] / scale)) for p in pts] if len(pts) == 2 else None


# ================= COMPENSAÇÃO DE ZOOM =================

def calculate_scale_factor_robust(pts_old, pts_new):
    if len(pts_old) < ZOOM_DETECTION_MIN_POINTS or len(pts_new) < ZOOM_DETECTION_MIN_POINTS:
        return 1.0

    pts_old_2d = pts_old.reshape(-1, 2)
    pts_new_2d = pts_new.reshape(-1, 2)
    n = min(len(pts_old_2d), len(pts_new_2d))

    all_ratios = []
    for i in range(n):
        for j in range(i + 1, n):
            d_old = np.linalg.norm(pts_old_2d[i] - pts_old_2d[j])
            d_new = np.linalg.norm(pts_new_2d[i] - pts_new_2d[j])
            if d_old > 5.0:
                ratio = d_new / d_old
                all_ratios.append(ratio)

    if len(all_ratios) == 0:
        return 1.0

    scale_factor = np.median(all_ratios)

    try:
        hull_old = cv2.convexHull(pts_old_2d.astype(np.float32))
        hull_new = cv2.convexHull(pts_new_2d.astype(np.float32))
        area_old = cv2.contourArea(hull_old)
        area_new = cv2.contourArea(hull_new)
        if area_old > 1.0:
            scale_from_area = np.sqrt(area_new / area_old)
            scale_factor = 0.7 * scale_factor + 0.3 * scale_from_area
    except:
        pass

    if abs(scale_factor - 1.0) < ZOOM_THRESHOLD:
        scale_factor = 1.0

    scale_factor = np.clip(scale_factor, 0.85, 1.15)
    return scale_factor


def calculate_centroid_with_weights(points, previous_centroid=None):
    if len(points) == 0:
        if previous_centroid is not None:
            return previous_centroid
        return np.array([0.0, 0.0])

    pts = points.reshape(-1, 2)
    centroid = np.median(pts, axis=0).astype(np.float32)

    if previous_centroid is not None:
        delta = np.linalg.norm(centroid - previous_centroid)
        if delta > 50:
            alpha = 0.3
            centroid = alpha * centroid + (1 - alpha) * previous_centroid

    return centroid


def compensate_zoom_movement_improved(centroid_old, centroid_new, scale_factor, image_center):
    if abs(scale_factor - 1.0) < ZOOM_THRESHOLD:
        return centroid_new - centroid_old

    apparent_movement = centroid_new - centroid_old
    vector_from_center_old = centroid_old - image_center
    expected_zoom_movement = vector_from_center_old * (scale_factor - 1.0)
    real_movement = apparent_movement - expected_zoom_movement
    return real_movement


def apply_kalman_filter(measurements, state, P, Q, R):
    state_pred = state
    P_pred = P + Q
    K = P_pred / (P_pred + R)
    state = state_pred + K * (measurements - state_pred)
    P = (1 - K) * P_pred
    return state, P


# ================= SELEÇÃO DE PONTOS =================

def select_points_interactive(frame, display_dim, scale, diametros_list, tempo_atual,
                              existing_pts_car=None, existing_pts_ref=None, force_diameter=False):
    img_disp = cv2.resize(frame, display_dim)
    gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    all_features = cv2.goodFeaturesToTrack(gray_init, 1000, 0.01, 10)
    if all_features is not None:
        all_features = all_features.reshape(-1, 2)
    else:
        all_features = np.array([])

    pts_car = list(existing_pts_car) if existing_pts_car is not None else []
    pts_ref = list(existing_pts_ref) if existing_pts_ref is not None else []

    mode = 0

    if force_diameter or tempo_atual == 0 or len(diametros_list) == 0:
        metodologia_text = "COM compensação" if METODOLOGIA == "com_zoom" else "SEM compensação (nova escala px/mm)"
        print(f"\n{'=' * 60}")
        print(f"METODOLOGIA: {metodologia_text}")
        print(f"{'=' * 60}")

        l_sup = select_line(img_disp, f"DIAMETRO: Linha SUPERIOR (Espaco p/ pular, ESC cancela)", scale)
        if l_sup is None:
            return None, None
        if l_sup:
            l_inf = select_line(img_disp, f"DIAMETRO: Linha INFERIOR (Espaco p/ pular, ESC cancela)", scale)
            if l_inf is None:
                return None, None
            if l_inf:
                diam = calculate_diameter(l_sup, l_inf)
                diametros_list.append([round(tempo_atual, 2), diam])
                print(f"✓ Diâmetro medido: {diam:.2f} px no tempo {tempo_atual:.2f} ms")

    mode = 1

    def mouse_callback(event, x, y, flags, param):
        nonlocal pts_car, pts_ref, mode

        if event == cv2.EVENT_LBUTTONDOWN:
            x_orig = int(x / scale)
            y_orig = int(y / scale)

            if mode == 1:
                removed = False
                for i, pt in enumerate(pts_car):
                    dist = np.linalg.norm(np.array([x_orig, y_orig]) - pt)
                    if dist < 15:
                        pts_car.pop(i)
                        print(f"Ponto CARRINHO removido")
                        removed = True
                        break
                if not removed and len(all_features) > 0:
                    dists = np.linalg.norm(all_features - np.array([x_orig, y_orig]), axis=1)
                    idx = np.argmin(dists)
                    if dists[idx] < 20:
                        new_pt = all_features[idx]
                        pts_car.append(new_pt)
                        print(f"Ponto CARRINHO adicionado ({len(pts_car)} total)")

            elif mode == 2:
                removed = False
                for i, pt in enumerate(pts_ref):
                    dist = np.linalg.norm(np.array([x_orig, y_orig]) - pt)
                    if dist < 15:
                        pts_ref.pop(i)
                        print(f"Ponto REFERÊNCIA removido")
                        removed = True
                        break
                if not removed and len(all_features) > 0:
                    dists = np.linalg.norm(all_features - np.array([x_orig, y_orig]), axis=1)
                    idx = np.argmin(dists)
                    if dists[idx] < 20:
                        new_pt = all_features[idx]
                        pts_ref.append(new_pt)
                        print(f"Ponto REFERÊNCIA adicionado ({len(pts_ref)} total)")

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("\n--- INSTRUÇÕES DE SELEÇÃO ---")
    print("1. Clique nas bolinhas AMARELAS para adicionar pontos")
    print("2. Clique em pontos JÁ SELECIONADOS para REMOVER")
    print("3. [TAB] Alternar entre CARRINHO (verde) e REFERÊNCIA (azul)")
    print("4. [ESPAÇO] Finalizar seleção")
    print("5. [ESC] Cancelar")

    while True:
        display = img_disp.copy()

        for f in all_features:
            x_disp, y_disp = int(f[0] * scale), int(f[1] * scale)
            cv2.circle(display, (x_disp, y_disp), 2, (0, 255, 255), -1)

        for pt in pts_car:
            x_disp, y_disp = int(pt[0] * scale), int(pt[1] * scale)
            cv2.circle(display, (x_disp, y_disp), 6, (0, 255, 0), -1)
            cv2.circle(display, (x_disp, y_disp), 8, (0, 255, 0), 2)

        for pt in pts_ref:
            x_disp, y_disp = int(pt[0] * scale), int(pt[1] * scale)
            cv2.circle(display, (x_disp, y_disp), 6, (255, 0, 0), -1)
            cv2.circle(display, (x_disp, y_disp), 8, (255, 0, 0), 2)

        cv2.rectangle(display, (0, 0), (display.shape[1], 120), (0, 0, 0), -1)

        mode_text = "CARRINHO (Verde)" if mode == 1 else "REFERÊNCIA (Azul)"
        if METODOLOGIA == "laboratorio":
            metodologia_tag = "LABORATÓRIO (sem ref.)"
        elif METODOLOGIA == "com_zoom":
            metodologia_tag = "COM ZOOM"
        else:
            metodologia_tag = "SEM ZOOM"
        cv2.putText(display, f"Modo: {mode_text} | Metodologia: {metodologia_tag}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if METODOLOGIA == "laboratorio":
            cv2.putText(display, f"Carrinho: {len(pts_car)} | (sem referência necessária)",
                        (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(display, f"Carrinho: {len(pts_car)} | Referência: {len(pts_ref)}",
                        (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "[TAB] Modo | [ESPACO] OK | [ESC] Cancelar",
                    (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if METODOLOGIA == "com_zoom" and len(pts_ref) < 6:
            cv2.putText(display, "Recomendado: 6+ pontos de REFERENCIA bem distribuidos",
                        (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        elif METODOLOGIA == "sem_zoom":
            cv2.putText(display, "Pontos REF medem zoom (sem negacao de movimento)",
                        (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
        elif METODOLOGIA == "laboratorio":
            cv2.putText(display, "Modo lab: apenas pontos do CARRINHO necessarios",
                        (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 150), 1)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            if len(pts_car) == 0:
                print("AVISO: É necessário ter pelo menos 1 ponto no CARRINHO!")
                continue
            if METODOLOGIA != "laboratorio" and len(pts_ref) == 0:
                print("AVISO: É necessário ter pelo menos 1 ponto de REFERÊNCIA!")
                continue
            break
        elif key == 9:
            if METODOLOGIA == "laboratorio":
                print("Modo laboratório: seleção de referência desativada.")
            else:
                mode = 2 if mode == 1 else 1
                print(f"\nModo alterado para: {'CARRINHO' if mode == 1 else 'REFERÊNCIA'}")
        elif key == 27:
            print("Seleção cancelada pelo usuário")
            cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)
            return None, None

    cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)

    pts_car_arr = np.array(pts_car, dtype=np.float32).reshape(-1, 1, 2)
    # No modo laboratório, pts_ref ficará vazio — retorna array vazio no formato correto
    if len(pts_ref) > 0:
        pts_ref_arr = np.array(pts_ref, dtype=np.float32).reshape(-1, 1, 2)
    else:
        pts_ref_arr = np.empty((0, 1, 2), dtype=np.float32)

    return pts_car_arr, pts_ref_arr


# ================= SELEÇÃO DE METODOLOGIA =================

# ================= EXECUÇÃO PRINCIPAL =================
Tk().withdraw()

# Derivar METODOLOGIA a partir das flags de configuração
if MODO_LABORATORIO:
    METODOLOGIA = "laboratorio"
elif USAR_COMPENSACAO_ZOOM:
    METODOLOGIA = "com_zoom"
else:
    METODOLOGIA = "sem_zoom"

_descricao_modo = {
    "laboratorio": "LABORATÓRIO (câmera fixa, sem referência, sem zoom)",
    "com_zoom":    "COM COMPENSAÇÃO DE ZOOM",
    "sem_zoom":    "SEM COMPENSAÇÃO DE ZOOM",
}
print(f"\n{'=' * 70}")
print(f"METODOLOGIA SELECIONADA: {_descricao_modo[METODOLOGIA]}")
print(f"{'=' * 70}\n")

video_path = filedialog.askopenfilename()
if not video_path:
    exit()

video_name = os.path.basename(video_path)
data_v, hora_v = extract_file_info(video_path)
last_start_frame = load_session(video_name)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Erro ao carregar vídeo.")
    exit()

h_orig, w_orig = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
image_center = np.array([w_orig / 2.0, h_orig / 2.0], dtype=np.float32)

scale = get_screen_scale(h_orig)
display_dim = (int(w_orig * scale), int(h_orig * scale))

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, display_dim[0], display_dim[1])
cap.set(cv2.CAP_PROP_POS_FRAMES, last_start_frame)

dados = []
diametros_medidos = []
posicao_xy_atual = np.array([0.0, 0.0], dtype=np.float32)

historico_pontos = {}
historico_zoom = {}

centroid_car_prev = None
centroid_ref_prev = None

zoom_state = 1.0
zoom_P = 1.0
zoom_Q = 0.0001
zoom_R = 0.01

# ============================================================
# ESTADO DO MODO PASSIVO
# ============================================================
modo_passivo = False          # True = rodando sem processar nada
frame_passivo_inicio = None   # Frame onde o modo passivo começou
dist_relativa = 0.0           # Inicializar para evitar NameError na visualização

# ================= NAVEGAÇÃO INICIAL =================
print("\n=== NAVEGAÇÃO INICIAL ===")
print("Posicione o vídeo no frame inicial desejado:")
print("  [D] Avançar 10 frames")
print("  [Shift+D] Avançar 100 frames")
print("  [A] Voltar 10 frames")
print("  [ESPAÇO] Confirmar e iniciar")
print("  [ESC] Cancelar e sair")

while True:
    curr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    preview = cv2.resize(frame, display_dim)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 80), (0, 0, 0), -1)

    metodologia_tag = {"laboratorio": "LABORATÓRIO", "com_zoom": "COM ZOOM", "sem_zoom": "SEM ZOOM"}.get(METODOLOGIA, METODOLOGIA)
    cv2.putText(preview, f"Metodologia: {metodologia_tag}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(preview, f"Frame {curr} | [A/D] +/-10 | [Shift+D] +100 | [ESPACO] Iniciar | [ESC] Sair",
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.imshow(WINDOW_NAME, preview)

    key = cv2.waitKey(0)
    if key == ord(' '):
        save_session(video_name, curr)
        break
    elif key == ord('d'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr + 10)
    elif key == ord('D'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr + 100)
    elif key == ord('a'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr - 10))
    elif key == 27:
        print("Operação cancelada pelo usuário")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# ================= SELEÇÃO INICIAL DE PONTOS =================
tempo_zero_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
frame_inicial = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
pts_car, pts_ref = select_points_interactive(frame, display_dim, scale, diametros_medidos, 0)

if pts_car is None or pts_ref is None:
    print("Seleção cancelada.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

historico_pontos[frame_inicial] = {
    'pts_car': pts_car.copy(),
    'pts_ref': pts_ref.copy(),
    'good_car': [pt[0] for pt in pts_car],
    'good_ref': [pt[0] for pt in pts_ref]
}

centroid_car_prev = calculate_centroid_with_weights(pts_car)
if len(pts_ref) > 0:
    centroid_ref_prev = calculate_centroid_with_weights(pts_ref)
else:
    centroid_ref_prev = centroid_car_prev.copy()

historico_zoom[frame_inicial] = 1.0
zoom_acumulado = 1.0
zoom_suavizado = 1.0

print(f"\n=== RASTREAMENTO INICIADO ===")
_desc_met = {"laboratorio": "LABORATÓRIO (câmera fixa, sem referência)", "com_zoom": "COM compensação de zoom", "sem_zoom": "SEM compensação de zoom"}
print(f"Metodologia: {_desc_met.get(METODOLOGIA, METODOLOGIA)}")
print(f"Pontos CARRINHO: {len(pts_car)}")
if METODOLOGIA != "laboratorio":
    print(f"Pontos REFERÊNCIA: {len(pts_ref)}")
print("\nControles durante rastreamento:")
print("  [M] Modo ajuste/recalibração")
print("  [ESC] ou [Q] Finalizar e salvar")

key_m_pressed = False

# ================= LOOP PRINCIPAL DE RASTREAMENTO =================
while True:
    curr_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ============================================================
    # MODO PASSIVO: apenas exibe o vídeo, sem processar odometria
    # ============================================================
    if modo_passivo:
        draw_passivo = cv2.resize(frame, display_dim)

        cv2.rectangle(draw_passivo, (0, 0), (draw_passivo.shape[1], 110), (0, 0, 0), -1)
        cv2.putText(draw_passivo,
                    f"MODO PASSIVO | Frame: {curr_idx}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
        cv2.putText(draw_passivo,
                    f"Nenhuma medicao sendo feita",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        cv2.putText(draw_passivo,
                    "[M] Retomar rastreamento com novos pontos | [ESC/Q] Sair",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

        # Borda vermelha para sinalizar modo passivo
        cv2.rectangle(draw_passivo, (0, 0),
                      (draw_passivo.shape[1] - 1, draw_passivo.shape[0] - 1),
                      (0, 0, 200), 4)

        cv2.imshow(WINDOW_NAME, draw_passivo)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('m'):
            # ---- Sair do modo passivo: ir para ajuste e selecionar novos pontos ----
            print(f"\n[MODO PASSIVO] Encerrando no frame {curr_idx}. Iniciando recalibração...")
            modo_passivo = False
            key_m_pressed = True   # força entrada no bloco de ajuste abaixo

        elif key == 27 or key == ord('q'):
            print("Finalizando rastreamento...")
            break

        # Atualiza prev_gray para que o OF funcione quando sair do modo passivo
        prev_gray = gray.copy()
        continue   # pula todo o restante do loop

    # ============================================================
    # PROCESSAMENTO NORMAL (modo ativo)
    # ============================================================

    pts_car_new, st_car, err_car = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_car, None, **lk_params)

    # No modo laboratório pts_ref é vazio — só calcula OF de referência se houver pontos
    if len(pts_ref) > 0:
        pts_ref_new, st_ref, err_ref = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_ref, None, **lk_params)
    else:
        pts_ref_new = np.empty((0, 1, 2), dtype=np.float32)
        st_ref = np.empty((0, 1), dtype=np.uint8)

    good_car = []
    good_car_indices = []
    for i, (new, found) in enumerate(zip(pts_car_new, st_car)):
        x, y = new.ravel()
        if found == 1 and (0 <= x < w) and (0 <= y < h):
            good_car.append(new)
            good_car_indices.append(i)

    good_ref = []
    good_ref_indices = []
    for i, (new, found) in enumerate(zip(pts_ref_new, st_ref)):
        x, y = new.ravel()
        if found == 1 and (0 <= x < w) and (0 <= y < h):
            good_ref.append(new)
            good_ref_indices.append(i)

    out_borda = False
    if len(good_car) > 0:
        for pt in good_car:
            x, y = pt.ravel()
            if x < SAFE_MARGIN or x > (w - SAFE_MARGIN):
                out_borda = True
                break

    if METODOLOGIA == "laboratorio":
        # No modo laboratório só recalibra por pontos do carrinho
        need_recalibration = (
            out_borda or
            len(good_car) < max(1, len(pts_car) * 0.3)
        )
    else:
        need_recalibration = (
            out_borda or
            len(good_car) < max(1, len(pts_car) * 0.3) or
            len(good_ref) < max(1, len(pts_ref) * 0.3)
        )

    key = cv2.waitKey(1) & 0xFF

    # ================= MODO AJUSTE/RECALIBRAÇÃO =================
    if key == ord('m') or need_recalibration or key_m_pressed:
        if need_recalibration:
            print(f"\n⚠️  RECALIBRAÇÃO AUTOMÁTICA (Frame {curr_idx})")
            print(f"   Carrinho: {len(good_car)}/{len(pts_car)} pontos")
            print(f"   Referência: {len(good_ref)}/{len(pts_ref)} pontos")

        key_m_pressed = False
        ajuste_frame_idx = curr_idx

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ajuste_frame_idx)
            ret, frame_adj = cap.read()
            if not ret:
                ajuste_frame_idx = max(0, ajuste_frame_idx - 1)
                continue

            t_atual = float(cap.get(cv2.CAP_PROP_POS_MSEC)) - tempo_zero_ms
            p_adj = cv2.resize(frame_adj, display_dim)

            hist_frames = sorted(historico_pontos.keys())
            hist_frame = None
            for hf in reversed(hist_frames):
                if hf <= ajuste_frame_idx:
                    hist_frame = hf
                    break

            if hist_frame is not None and hist_frame in historico_pontos:
                hist_data = historico_pontos[hist_frame]
                if 'good_car' in hist_data and len(hist_data['good_car']) > 0:
                    for pt in hist_data['good_car']:
                        x, y = pt[0], pt[1]
                        cv2.circle(p_adj, (int(x * scale), int(y * scale)), 5, (0, 255, 0), -1)
                if 'good_ref' in hist_data and len(hist_data['good_ref']) > 0:
                    for pt in hist_data['good_ref']:
                        x, y = pt[0], pt[1]
                        cv2.circle(p_adj, (int(x * scale), int(y * scale)), 5, (255, 0, 0), -1)
                num_car = len(hist_data['good_car']) if 'good_car' in hist_data else 0
                num_ref = len(hist_data['good_ref']) if 'good_ref' in hist_data else 0
            else:
                num_car = len(good_car)
                num_ref = len(good_ref)

            cv2.rectangle(p_adj, (0, 0), (p_adj.shape[1], 175), (0, 0, 0), -1)
            cv2.putText(p_adj, f"AJUSTE Frame {ajuste_frame_idx}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(p_adj,
                        "[A/D] Navegar | [ESPACO] Recalibrar | [C] Continuar | [ESC] Sair",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(p_adj, f"Car: {num_car} | Ref: {num_ref}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # ---- INSTRUÇÃO DO MODO PASSIVO ----
            cv2.putText(p_adj,
                        "[N] Modo PASSIVO: avanca sem processar ate proximo [M]",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)
            cv2.putText(p_adj,
                        "  -> Apague dados futuros, navegue ate o ponto certo e pressione [N]",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 255), 1)

            if hist_frame is not None and hist_frame in historico_zoom:
                zoom_val = historico_zoom[hist_frame]
                cv2.putText(p_adj, f"Zoom: {zoom_val:.4f}x",
                            (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            metodologia_tag = "COM ZOOM" if METODOLOGIA == "com_zoom" else "SEM ZOOM + NOVA ESCALA"
            cv2.putText(p_adj, f"Metodologia: {metodologia_tag}",
                        (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

            cv2.imshow(WINDOW_NAME, p_adj)

            k2 = cv2.waitKey(0) & 0xFF
            if k2 == ord('a'):
                ajuste_frame_idx = max(0, ajuste_frame_idx - 5)
            elif k2 == ord('d'):
                ajuste_frame_idx += 5
            elif k2 == ord(' '):
                curr_idx = ajuste_frame_idx
                break
            elif k2 == ord('c'):
                if len(good_car) > 0 and len(good_ref) > 0:
                    pts_car = np.array(good_car, dtype=np.float32).reshape(-1, 1, 2)
                    pts_ref = np.array(good_ref, dtype=np.float32).reshape(-1, 1, 2)
                    curr_idx = ajuste_frame_idx
                    break
                else:
                    print("Não é possível continuar sem pontos válidos!")
            elif k2 == ord('m'):
                key_m_pressed = True
            elif k2 == ord('n'):
                # =====================================================
                # [N] ATIVAR MODO PASSIVO
                # =====================================================
                print(f"\n{'=' * 60}")
                print(f"[N] MODO PASSIVO ATIVADO a partir do frame {ajuste_frame_idx}")
                print(f"  -> Dados futuros a partir do frame {ajuste_frame_idx} serão apagados")
                print(f"  -> Vídeo avançará SEM processar odometria")
                print(f"  -> Pressione [M] no loop principal para retomar rastreamento")
                print(f"{'=' * 60}")

                # Apagar todos os dados (odometria e histórico) a partir deste frame
                t_corte_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC)) - tempo_zero_ms
                dados = [d for d in dados if d[0] < t_corte_ms]

                frames_to_remove = [f for f in historico_pontos.keys() if f >= ajuste_frame_idx]
                for f in frames_to_remove:
                    del historico_pontos[f]
                    if f in historico_zoom:
                        del historico_zoom[f]

                # Restaurar zoom e posição do último ponto do histórico válido
                hist_frames_restantes = sorted(historico_pontos.keys())
                if hist_frames_restantes:
                    last_valid = hist_frames_restantes[-1]
                    if last_valid in historico_zoom:
                        zoom_acumulado = historico_zoom[last_valid]
                        zoom_suavizado = zoom_acumulado
                        zoom_state = zoom_acumulado
                        zoom_P = 1.0

                # Reposicionar o vídeo no frame de início do modo passivo
                cap.set(cv2.CAP_PROP_POS_FRAMES, ajuste_frame_idx)
                ret, frame_passivo = cap.read()
                if ret:
                    prev_gray = cv2.cvtColor(frame_passivo, cv2.COLOR_BGR2GRAY)

                frame_passivo_inicio = ajuste_frame_idx
                modo_passivo = True

                # Sair do loop de ajuste sem fazer mais nada
                k2 = 0  # valor sentinela para não cair nos outros ifs
                break

            elif k2 == 27 or k2 == ord('q'):
                print("Finalizando rastreamento...")
                break

        # Verificar se saiu do loop de ajuste por ESC/Q
        if k2 == 27 or k2 == ord('q'):
            break

        # Se entrou em modo passivo, volta ao loop principal sem mais processamento
        if modo_passivo:
            continue

        if k2 == ord(' '):
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_idx)
            ret, frame = cap.read()

            tempo_ajuste_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC)) - tempo_zero_ms
            dados = [d for d in dados if d[0] < tempo_ajuste_ms]

            frames_to_remove = [f for f in historico_pontos.keys() if f >= curr_idx]
            for f in frames_to_remove:
                del historico_pontos[f]
                if f in historico_zoom:
                    del historico_zoom[f]

            hist_frames = sorted(historico_pontos.keys())
            existing_car = None
            existing_ref = None

            if len(hist_frames) > 0:
                last_hist_frame = hist_frames[-1]
                if last_hist_frame in historico_pontos:
                    hist_data = historico_pontos[last_hist_frame]
                    if 'good_car' in hist_data:
                        existing_car = hist_data['good_car']
                    if 'good_ref' in hist_data:
                        existing_ref = hist_data['good_ref']
                if last_hist_frame in historico_zoom:
                    zoom_acumulado = historico_zoom[last_hist_frame]
                    zoom_suavizado = zoom_acumulado
                    zoom_state = zoom_acumulado
                    zoom_P = 1.0

            force_diameter = (METODOLOGIA == "sem_zoom")

            pts_car, pts_ref = select_points_interactive(
                frame, display_dim, scale, diametros_medidos, tempo_ajuste_ms,
                existing_car, existing_ref, force_diameter
            )

            if pts_car is None or pts_ref is None:
                print("Recalibração cancelada, encerrando...")
                break

            historico_pontos[curr_idx] = {
                'pts_car': pts_car.copy(),
                'pts_ref': pts_ref.copy(),
                'good_car': [pt[0] for pt in pts_car],
                'good_ref': [pt[0] for pt in pts_ref]
            }

            historico_zoom[curr_idx] = zoom_acumulado
            centroid_car_prev = calculate_centroid_with_weights(pts_car)
            if len(pts_ref) > 0:
                centroid_ref_prev = calculate_centroid_with_weights(pts_ref)
            else:
                centroid_ref_prev = centroid_car_prev.copy()
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(f"✓ Recalibração concluída (Frame {curr_idx})")

        continue

    # ================= TECLA ESC NO LOOP PRINCIPAL =================
    if key == 27 or key == ord('q'):
        print("Finalizando rastreamento...")
        break

    # ================= CÁLCULO DE ODOMETRIA =================
    # No modo laboratório: só precisa de pontos do carrinho
    _tem_ref = len(good_ref) > 0
    _pode_calcular = len(good_car) > 0 and (METODOLOGIA == "laboratorio" or _tem_ref)

    if _pode_calcular:
        pts_car_new_arr = np.array(good_car, dtype=np.float32).reshape(-1, 1, 2)
        pts_car_old_matched = pts_car[good_car_indices]

        if _tem_ref:
            pts_ref_new_arr = np.array(good_ref, dtype=np.float32).reshape(-1, 1, 2)
            pts_ref_old_matched = pts_ref[good_ref_indices]
            scale_factor_raw = calculate_scale_factor_robust(pts_ref_old_matched, pts_ref_new_arr)
        else:
            pts_ref_new_arr = np.empty((0, 1, 2), dtype=np.float32)
            pts_ref_old_matched = np.empty((0, 1, 2), dtype=np.float32)
            scale_factor_raw = 1.0

        if METODOLOGIA == "com_zoom":
            zoom_state, zoom_P = apply_kalman_filter(scale_factor_raw, zoom_state, zoom_P, zoom_Q, zoom_R)
            scale_factor = zoom_state
            zoom_suavizado = ZOOM_SMOOTH_ALPHA * scale_factor + (1 - ZOOM_SMOOTH_ALPHA) * zoom_suavizado
            scale_factor_final = zoom_suavizado
            if abs(scale_factor_final - 1.0) > ZOOM_THRESHOLD:
                zoom_acumulado *= scale_factor_final
        else:
            scale_factor_final = 1.0
            zoom_acumulado = 1.0

        historico_zoom[curr_idx] = zoom_acumulado

        centroid_car_old = calculate_centroid_with_weights(pts_car_old_matched, centroid_car_prev)
        centroid_car_new = calculate_centroid_with_weights(pts_car_new_arr, centroid_car_old)
        centroid_car_prev = centroid_car_new.copy()

        if METODOLOGIA == "laboratorio":
            # Câmera fixa: deslocamento do carrinho é direto, sem subtração de câmera
            mov_real = centroid_car_new - centroid_car_old
            centroid_ref_new = centroid_car_new  # sentinela para não quebrar visualização
        else:
            centroid_ref_old = calculate_centroid_with_weights(pts_ref_old_matched, centroid_ref_prev)
            centroid_ref_new = calculate_centroid_with_weights(pts_ref_new_arr, centroid_ref_old)
            centroid_ref_prev = centroid_ref_new.copy()

            mov_camera = compensate_zoom_movement_improved(
                centroid_ref_old, centroid_ref_new, scale_factor_final, image_center
            )
            mov_car = compensate_zoom_movement_improved(
                centroid_car_old, centroid_car_new, scale_factor_final, image_center
            )
            mov_real = mov_car - mov_camera

        zoom_negado = False
        # Negação de zoom removida: sem câmera móvel não faz sentido negar movimento
        # (bloco mantido vazio para referência futura)

        if np.linalg.norm(mov_real) < 0.5:
            mov_real = np.array([0.0, 0.0], dtype=np.float32)

        posicao_xy_atual += mov_real
        dist_relativa = np.linalg.norm(posicao_xy_atual)

        t_rel = round(float(cap.get(cv2.CAP_PROP_POS_MSEC)) - tempo_zero_ms, 2)

        dados.append([
            t_rel,
            round(float(dist_relativa), 2),
            float(centroid_car_new[0]),
            float(centroid_car_new[1]),
            len(good_car),
            float(centroid_ref_new[0]),
            float(centroid_ref_new[1]),
            len(good_ref),
            float(scale_factor_raw),
            float(scale_factor_final),
            float(zoom_acumulado),
            zoom_negado if METODOLOGIA == "sem_zoom" else False
        ])

        historico_pontos[curr_idx] = {
            'pts_car': pts_car_new_arr.copy(),
            'pts_ref': pts_ref_new_arr.copy(),
            'good_car': [pt[0] for pt in pts_car_new_arr],
            'good_ref': [pt[0] for pt in pts_ref_new_arr]
        }

        pts_car = pts_car_new_arr
        pts_ref = pts_ref_new_arr

    # ================= VISUALIZAÇÃO =================
    if curr_idx % FAST_FORWARD == 0:
        draw = frame.copy()

        if len(good_car) > 0:
            for pt in good_car:
                x, y = pt.ravel()
                cv2.circle(draw, (int(x), int(y)), 4, (0, 255, 0), -1)
            centroid = calculate_centroid_with_weights(pts_car_new_arr)
            cv2.circle(draw, (int(centroid[0]), int(centroid[1])), 8, (0, 255, 0), 2)

        if len(good_ref) > 0:
            for pt in good_ref:
                x, y = pt.ravel()
                cv2.circle(draw, (int(x), int(y)), 4, (255, 0, 0), -1)
            centroid = calculate_centroid_with_weights(pts_ref_new_arr)
            cv2.circle(draw, (int(centroid[0]), int(centroid[1])), 8, (255, 0, 0), 2)

        res_v = cv2.resize(draw, display_dim)

        cv2.rectangle(res_v, (0, 0), (res_v.shape[1], 150), (0, 0, 0), -1)
        cv2.putText(res_v, f"Distancia: {dist_relativa:.1f}px | Frame: {curr_idx}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(res_v, f"Carrinho: {len(good_car)} pts | Ref: {len(good_ref)} pts",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        metodologia_tag = {
            "laboratorio": "LABORATÓRIO",
            "com_zoom":    "COM ZOOM",
            "sem_zoom":    "SEM ZOOM",
        }.get(METODOLOGIA, METODOLOGIA)
        cv2.putText(res_v, f"Metodologia: {metodologia_tag}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

        if METODOLOGIA == "com_zoom":
            zoom_color = (0, 255, 255)
            if abs(scale_factor_final - 1.0) > 0.01:
                zoom_color = (0, 165, 255)
            cv2.putText(res_v, f"Zoom Frame: {scale_factor_final:.4f}x | Acum: {zoom_acumulado:.4f}x",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, zoom_color, 1)
            if abs(scale_factor_final - 1.0) > ZOOM_THRESHOLD:
                cv2.putText(res_v, "COMPENSANDO ZOOM",
                            (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        elif METODOLOGIA == "laboratorio":
            cv2.putText(res_v, "Camera fixa | Deslocamento direto do carrinho",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 150), 1)
        else:
            cv2.putText(res_v, f"Zoom: {scale_factor_raw:.4f}x (fixado em 1.0x, sem negacao)",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(res_v,
                    "[M] Ajuste" + (" + Nova Escala" if METODOLOGIA == "sem_zoom" else "") + " | [ESC/Q] Salvar",
                    (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME, res_v)

    prev_gray = gray.copy()

# ================= SALVAR RESULTADOS =================
print("\n=== SALVANDO RESULTADOS ===")

historico_serializado = {}
for frame_idx, data in historico_pontos.items():
    historico_serializado[str(frame_idx)] = {
        'good_car': [[float(pt[0]), float(pt[1])] for pt in data['good_car']],
        'good_ref': [[float(pt[0]), float(pt[1])] for pt in data['good_ref']]
    }

zoom_serializado = {str(k): float(v) for k, v in historico_zoom.items()}

resultado = {
    "video": video_name,
    "data_video": data_v,
    "hora_video": hora_v,
    "fps": fps,
    "frame_inicial": frame_inicial,
    "metodologia": METODOLOGIA,
    "diametros": diametros_medidos,
    "odometria": [[d[0], d[1]] for d in dados],
    "dados_completos": dados,
    "historico_pontos": historico_serializado,
    "historico_zoom": zoom_serializado,
    "zoom_final": float(zoom_acumulado),
    "total_frames_rastreados": len(historico_pontos),
    "parametros_zoom": {
        "threshold": ZOOM_THRESHOLD,
        "smooth_alpha": ZOOM_SMOOTH_ALPHA if METODOLOGIA == "com_zoom" else "N/A",
        "min_points": ZOOM_DETECTION_MIN_POINTS,
        "kalman_Q": zoom_Q if METODOLOGIA == "com_zoom" else "N/A",
        "kalman_R": zoom_R if METODOLOGIA == "com_zoom" else "N/A"
    }
}

metodologia_sufixo = METODOLOGIA  # "laboratorio", "com_zoom" ou "sem_zoom"
output_file = video_path + f"_odometria_OF_{metodologia_sufixo}.json"
with open(output_file, "w") as f:
    json.dump(resultado, f, indent=4)

print(f"✓ Arquivo salvo: {output_file}")
print(f"✓ Metodologia: {METODOLOGIA}")
print(f"✓ Total de medições: {len(dados)}")
print(f"✓ Medições de diâmetro: {len(diametros_medidos)}")
print(f"✓ Frames com histórico: {len(historico_pontos)}")

if len(dados) > 0:
    print(f"✓ Distância final: {dist_relativa:.2f} pixels")
else:
    print("✓ Nenhuma medição realizada.")

if METODOLOGIA == "com_zoom":
    print(f"✓ Zoom acumulado: {zoom_acumulado:.4f}x")
    frames_com_zoom = sum(1 for d in dados if abs(d[9] - 1.0) > ZOOM_THRESHOLD)
    print(f"✓ Frames com zoom detectado: {frames_com_zoom}")
elif METODOLOGIA == "laboratorio":
    print(f"✓ Modo laboratório: câmera fixa, deslocamento direto do carrinho.")
    print(f"✓ Zoom desativado: nenhuma negação de movimento aplicada.")
else:
    print(f"✓ Zoom sempre fixado em: 1.0x (sem negação de movimento)")

cap.release()
cv2.destroyAllWindows()