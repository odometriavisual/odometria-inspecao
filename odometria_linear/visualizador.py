import cv2
import numpy as np
import json
from tkinter import Tk, filedialog
import os

GRAPH_HEIGHT = 300
OUTPUT_FPS   = 30
COLOR        = (0, 200, 0)   # verde


def load_odometria(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    odometria = data.get('odometria', [])
    fps       = data.get('fps', 30.0)

    frame_inicial = 0
    if 'historico_pontos' in data and data['historico_pontos']:
        frames = [int(k) for k in data['historico_pontos'].keys()]
        if frames:
            frame_inicial = min(frames)

    dados_frame = []
    for tempo_ms, dist_px in odometria:
        frame_abs = frame_inicial + int((tempo_ms / 1000.0) * fps)
        dados_frame.append([frame_abs, dist_px])

    print(f"  {len(dados_frame)} pontos | frames {dados_frame[0][0]}–{dados_frame[-1][0]}")
    return dados_frame, frame_inicial


def interpolate(dados, frame):
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


def draw_graph(width, height, dados, current_frame, max_dist, frame_inicial):
    graph = np.ones((height, width, 3), dtype=np.uint8) * 255

    ml, mr, mt, mb = 60, 20, 30, 40
    gw = width - ml - mr
    gh = height - mt - mb

    cv2.rectangle(graph, (ml, mt), (ml + gw, mt + gh), (240, 240, 240), -1)

    for i in range(6):
        y = mt + int(i * gh / 5)
        cv2.line(graph, (ml, y), (ml + gw, y), (200, 200, 200), 1)
        cv2.putText(graph, f"{int(max_dist * (1 - i / 5))}",
                    (5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    max_frame  = dados[-1][0] if dados else frame_inicial + 1000
    frame_range = max(max_frame - frame_inicial, 1)

    for i in range(11):
        x = ml + int(i * gw / 10)
        cv2.line(graph, (x, mt), (x, mt + gh), (200, 200, 200), 1)
        fv = frame_inicial + int(i * frame_range / 10)
        cv2.putText(graph, f"{fv}", (x - 20, mt + gh + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    def fx(f):
        return ml + int(((f - frame_inicial) / frame_range) * gw)

    def fy(d):
        return mt + int((1 - d / max_dist) * gh)

    if len(dados) > 1:
        pts = np.array([[fx(f), fy(d)] for f, d in dados], dtype=np.int32)
        cv2.polylines(graph, [pts], False, COLOR, 2)

    cx = fx(current_frame)
    cv2.line(graph, (cx, mt), (cx, mt + gh), (255, 0, 0), 2)

    dist = interpolate(dados, current_frame)
    cy   = fy(dist)
    cv2.circle(graph, (cx, cy), 6, COLOR, -1)
    darker = tuple(int(c * 0.7) for c in COLOR)
    cv2.circle(graph, (cx, cy), 8, darker, 2)

    cv2.rectangle(graph, (0, 0), (width, mt), (50, 50, 50), -1)
    cv2.putText(graph, "Deslocamento (px)",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(graph, "Distancia (px)",
                (5, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(graph, "Frame",
                (width // 2 - 20, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return graph


# ================= EXECUÇÃO PRINCIPAL =================
print("=== GERADOR DE VÍDEO COM GRÁFICO DE DESLOCAMENTO ===\n")

Tk().withdraw()

print("Selecione o arquivo de odometria (JSON):")
json_path = filedialog.askopenfilename(title="Odometria", filetypes=[("JSON", "*.json")])
if not json_path:
    exit()

print("Selecione o vídeo:")
video_path = filedialog.askopenfilename(title="Vídeo", filetypes=[("Video", "*.mp4 *.avi *.mov")])
if not video_path:
    exit()

print("Onde salvar o vídeo de saída:")
output_path = filedialog.asksaveasfilename(
    title="Salvar como", defaultextension=".mp4",
    filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")]
)
if not output_path:
    exit()

print("\nCarregando odometria...")
dados, frame_inicial = load_odometria(json_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Erro ao abrir vídeo.")
    exit()

h, w      = frame.shape[:2]
fps       = cap.get(cv2.CAP_PROP_FPS)
total_fr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
max_dist  = max(d[1] for d in dados) * 1.1

print(f"Vídeo: {w}x{h} @ {fps:.1f} FPS | {total_fr} frames")
print(f"Distância máxima: {max_dist:.1f} px\n")

output_h = h + GRAPH_HEIGHT + 60
fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
out      = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (w, output_h))

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicial)
frames_proc = 0
total_proc  = total_fr - frame_inicial

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    dist    = interpolate(dados, current)

    # Overlay no vídeo
    cv2.rectangle(frame, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame: {current}/{total_fr}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Deslocamento: {dist:.1f} px",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                tuple(min(255, int(c * 1.3)) for c in COLOR), 2)

    graph = draw_graph(w, GRAPH_HEIGHT, dados, current, max_dist, frame_inicial)

    # Painel inferior com valor atual
    panel = np.ones((60, w, 3), dtype=np.uint8) * 40
    cv2.rectangle(panel, (10, 10), (30, 30), COLOR, -1)
    name = os.path.basename(json_path)[-40:]
    cv2.putText(panel, name, (40, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel, f"{dist:.1f} px", (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                tuple(min(255, int(c * 1.3)) for c in COLOR), 2)

    out.write(np.vstack([frame, graph, panel]))
    frames_proc += 1

    if frames_proc % 30 == 0:
        print(f"Progresso: {frames_proc / total_proc * 100:.1f}% ({frames_proc}/{total_proc})")

cap.release()
out.release()

print(f"\n✓ Vídeo salvo em: {output_path}")
print(f"✓ Frames processados: {frames_proc}")
print(f"✓ Duração: {frames_proc / OUTPUT_FPS:.1f} s")
