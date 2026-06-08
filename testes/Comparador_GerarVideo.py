import cv2
import numpy as np
import json
from tkinter import Tk, filedialog
import os

# ================= CONFIGURAÇÕES =================
GRAPH_HEIGHT = 300  # Altura do gráfico
MARGIN = 50  # Margem para legendas
OUTPUT_FPS = 30  # FPS do vídeo de saída


# ================= FUNÇÕES DE APOIO =================

def load_odometria(json_path):
    """Carrega arquivo JSON de odometria"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extrair dados de odometria [tempo_ms, distancia_px]
        odometria = data.get('odometria', [])
        fps = data.get('fps', 30.0)

        # Tentar encontrar frame inicial
        frame_inicial = 0

        # Método 1: Ver se tem histórico de pontos (optical flow)
        if 'historico_pontos' in data and data['historico_pontos']:
            frames_hist = [int(k) for k in data['historico_pontos'].keys()]
            if frames_hist:
                frame_inicial = min(frames_hist)
                print(f"  Frame inicial (do histórico): {frame_inicial}")

        # Método 2: Calcular a partir do primeiro tempo de odometria
        if frame_inicial == 0 and odometria and len(odometria) > 0:
            # O primeiro registro pode ter tempo > 0 se começou depois
            primeiro_tempo_ms = odometria[0][0]
            frame_inicial = int((primeiro_tempo_ms / 1000.0) * fps)
            if frame_inicial > 0:
                print(f"  Frame inicial (calculado do tempo): {frame_inicial}")

        # Converter tempo relativo para frame absoluto
        dados_frame = []
        for tempo_ms, dist_px in odometria:
            # tempo_ms é relativo ao início da odometria
            frame_relativo = int((tempo_ms / 1000.0) * fps)
            frame_absoluto = frame_inicial + frame_relativo
            dados_frame.append([frame_absoluto, dist_px])

        return dados_frame, data, frame_inicial
    except Exception as e:
        print(f"Erro ao carregar {json_path}: {e}")
        return None, None, 0


def interpolate_distance(dados, frame):
    """Interpola distância para um frame específico"""
    if not dados or len(dados) == 0:
        return 0.0

    # Se frame está antes do primeiro registro
    if frame <= dados[0][0]:
        return dados[0][1]

    # Se frame está depois do último registro
    if frame >= dados[-1][0]:
        return dados[-1][1]

    # Interpolação linear entre dois pontos
    for i in range(len(dados) - 1):
        frame1, dist1 = dados[i]
        frame2, dist2 = dados[i + 1]

        if frame1 <= frame <= frame2:
            # Interpolação linear
            t = (frame - frame1) / (frame2 - frame1)
            return dist1 + t * (dist2 - dist1)

    return dados[-1][1]


def draw_graph(width, height, dados_a, dados_b, current_frame, max_dist, frame_inicial):
    """Desenha gráfico de comparação das odometrias"""
    graph = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Margens
    margin_left = 60
    margin_right = 20
    margin_top = 30
    margin_bottom = 40

    graph_w = width - margin_left - margin_right
    graph_h = height - margin_top - margin_bottom

    # Fundo do gráfico
    cv2.rectangle(graph,
                  (margin_left, margin_top),
                  (margin_left + graph_w, margin_top + graph_h),
                  (240, 240, 240), -1)

    # Grade horizontal
    num_lines = 5
    for i in range(num_lines + 1):
        y = margin_top + int(i * graph_h / num_lines)
        cv2.line(graph, (margin_left, y), (margin_left + graph_w, y), (200, 200, 200), 1)

        # Legendas do eixo Y (distância)    
        dist_value = max_dist * (1 - i / num_lines)
        cv2.putText(graph, f"{int(dist_value)}",
                    (5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Encontrar frame máximo (já em valores absolutos)
    max_frame = frame_inicial
    if dados_a and len(dados_a) > 0:
        max_frame = max(max_frame, dados_a[-1][0])
    if dados_b and len(dados_b) > 0:
        max_frame = max(max_frame, dados_b[-1][0])

    if max_frame == frame_inicial:
        max_frame = frame_inicial + 1000

    # Grade vertical (tempo/frames) - com offset
    num_v_lines = 10
    frame_range = max_frame - frame_inicial
    for i in range(num_v_lines + 1):
        x = margin_left + int(i * graph_w / num_v_lines)
        cv2.line(graph, (x, margin_top), (x, margin_top + graph_h), (200, 200, 200), 1)

        # Legendas do eixo X (frames absolutos)
        frame_value = frame_inicial + int(i * frame_range / num_v_lines)
        cv2.putText(graph, f"{frame_value}",
                    (x - 20, margin_top + graph_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # Função para converter coordenadas (considerando offset)
    def frame_to_x(frame):
        if frame_range == 0:
            return margin_left
        return margin_left + int(((frame - frame_inicial) / frame_range) * graph_w)

    def dist_to_y(dist):
        return margin_top + int((1 - dist / max_dist) * graph_h)

    # Desenhar linha da Odometria A (VERDE)
    if dados_a and len(dados_a) > 1:
        pts_a = []
        for frame, dist in dados_a:
            x = frame_to_x(frame)
            y = dist_to_y(dist)
            pts_a.append([x, y])

        pts_a = np.array(pts_a, dtype=np.int32)
        cv2.polylines(graph, [pts_a], False, (0, 200, 0), 2)

    # Desenhar linha da Odometria B (AZUL)
    if dados_b and len(dados_b) > 1:
        pts_b = []
        for frame, dist in dados_b:
            x = frame_to_x(frame)
            y = dist_to_y(dist)
            pts_b.append([x, y])

        pts_b = np.array(pts_b, dtype=np.int32)
        cv2.polylines(graph, [pts_b], False, (200, 0, 0), 2)

    # Linha vertical do frame atual
    current_x = frame_to_x(current_frame)
    cv2.line(graph, (current_x, margin_top), (current_x, margin_top + graph_h),
             (255, 0, 0), 2)

    # Círculos nos pontos atuais
    dist_a = interpolate_distance(dados_a, current_frame)
    dist_b = interpolate_distance(dados_b, current_frame)

    if dados_a:
        cv2.circle(graph, (current_x, dist_to_y(dist_a)), 6, (0, 200, 0), -1)
        cv2.circle(graph, (current_x, dist_to_y(dist_a)), 8, (0, 150, 0), 2)

    if dados_b:
        cv2.circle(graph, (current_x, dist_to_y(dist_b)), 6, (200, 0, 0), -1)
        cv2.circle(graph, (current_x, dist_to_y(dist_b)), 8, (150, 0, 0), 2)

    # Legendas
    cv2.rectangle(graph, (0, 0), (width, margin_top), (50, 50, 50), -1)
    cv2.putText(graph, "Comparacao de Odometrias",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Título dos eixos
    cv2.putText(graph, "Distancia (px)",
                (5, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(graph, "Frame",
                (width // 2 - 20, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return graph


def draw_info_panel(width, dist_a, dist_b, frame, name_a, name_b):
    """Desenha painel de informações"""
    panel = np.ones((100, width, 3), dtype=np.uint8) * 40

    # Odometria A (VERDE)
    cv2.rectangle(panel, (10, 10), (30, 30), (0, 200, 0), -1)
    cv2.putText(panel, f"A: {name_a}",
                (40, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel, f"{dist_a:.1f} px",
                (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Odometria B (AZUL)
    cv2.rectangle(panel, (10, 55), (30, 75), (200, 0, 0), -1)
    cv2.putText(panel, f"B: {name_b}",
                (40, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel, f"{dist_b:.1f} px",
                (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

    # Diferença
    diff = abs(dist_a - dist_b)
    color_diff = (0, 255, 255) if diff < 10 else (0, 165, 255) if diff < 50 else (0, 0, 255)
    cv2.putText(panel, f"Diferenca: {diff:.1f} px ({diff / max(dist_a, dist_b, 1) * 100:.1f}%)",
                (width // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_diff, 2)

    # Frame atual
    cv2.putText(panel, f"Frame: {frame}",
                (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return panel


# ================= EXECUÇÃO PRINCIPAL =================
print("=== GERADOR DE VÍDEO COMPARATIVO ===\n")

# Seleção de arquivos
Tk().withdraw()

print("Selecione o arquivo de Odometria A (VERDE):")
path_odom_a = filedialog.askopenfilename(title="Odometria A", filetypes=[("JSON", "*.json")])
if not path_odom_a:
    print("Nenhum arquivo selecionado.")
    exit()

print("Selecione o arquivo de Odometria B (AZUL):")
path_odom_b = filedialog.askopenfilename(title="Odometria B", filetypes=[("JSON", "*.json")])
if not path_odom_b:
    print("Nenhum arquivo selecionado.")
    exit()

print("Selecione o arquivo de vídeo:")
video_path = filedialog.askopenfilename(title="Vídeo", filetypes=[("Video", "*.mp4 *.avi *.mov")])
if not video_path:
    print("Nenhum arquivo selecionado.")
    exit()

# Perguntar onde salvar o vídeo de saída
print("Escolha onde salvar o vídeo comparativo:")
output_path = filedialog.asksaveasfilename(
    title="Salvar vídeo como",
    defaultextension=".mp4",
    filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")]
)
if not output_path:
    print("Nenhum destino selecionado.")
    exit()

# Carregar odometrias
print("\nCarregando odometrias...")
dados_a, info_a, frame_inicial_a = load_odometria(path_odom_a)
dados_b, info_b, frame_inicial_b = load_odometria(path_odom_b)

if dados_a is None or dados_b is None:
    print("Erro ao carregar arquivos de odometria.")
    exit()

# Usar o menor frame inicial como ponto de partida
frame_inicial = min(frame_inicial_a, frame_inicial_b)

name_a = os.path.basename(path_odom_a).replace("_odometria_OF.json", "").replace("_odometria.json", "")[-30:]
name_b = os.path.basename(path_odom_b).replace("_odometria_OF.json", "").replace("_odometria.json", "")[-30:]

print(f"✓ Odometria A: {len(dados_a)} pontos (início: frame {frame_inicial_a})")
print(f"✓ Odometria B: {len(dados_b)} pontos (início: frame {frame_inicial_b})")
print(f"✓ Frame inicial de comparação: {frame_inicial}")

# Abrir vídeo
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Erro ao abrir vídeo.")
    exit()

h_orig, w_orig = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"✓ Vídeo: {w_orig}x{h_orig} @ {fps:.1f} FPS ({total_frames} frames)")

# Usar resolução original do vídeo
video_w = w_orig
video_h = h_orig

# Encontrar distância máxima para escala do gráfico
max_dist_a = max([d[1] for d in dados_a]) if dados_a else 0
max_dist_b = max([d[1] for d in dados_b]) if dados_b else 0
max_dist = max(max_dist_a, max_dist_b) * 1.1  # 10% de margem

print(f"\nDistância máxima: {max_dist:.1f} px")

# Configurar VideoWriter
output_w = video_w
output_h = video_h + GRAPH_HEIGHT + 100

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (output_w, output_h))

if not out.isOpened():
    print("Erro ao criar arquivo de vídeo.")
    exit()

print(f"\nGerando vídeo: {output_w}x{output_h} @ {OUTPUT_FPS} FPS")
print(f"Processando do frame {frame_inicial} ao {total_frames}...")

# Voltar ao frame inicial
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicial)

# Processar vídeo
frames_processados = 0
total_a_processar = total_frames - frame_inicial

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    # Obter distâncias atuais
    dist_a = interpolate_distance(dados_a, current_frame)
    dist_b = interpolate_distance(dados_b, current_frame)

    # Desenhar informações no vídeo
    cv2.rectangle(frame, (0, 0), (video_w, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"Frame: {current_frame}/{total_frames}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"A: {dist_a:.1f}px",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"B: {dist_b:.1f}px",
                (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)

    diff = abs(dist_a - dist_b)
    cv2.putText(frame, f"Diff: {diff:.1f}px",
                (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Desenhar gráfico
    graph = draw_graph(video_w, GRAPH_HEIGHT, dados_a, dados_b, current_frame, max_dist, frame_inicial)

    # Desenhar painel de informações
    info_panel = draw_info_panel(video_w, dist_a, dist_b, current_frame, name_a, name_b)

    # Combinar tudo verticalmente
    combined = np.vstack([frame, graph, info_panel])

    # Escrever frame no vídeo de saída
    out.write(combined)

    frames_processados += 1

    # Mostrar progresso
    if frames_processados % 30 == 0:
        progresso = (frames_processados / total_a_processar) * 100
        print(f"Progresso: {progresso:.1f}% ({frames_processados}/{total_a_processar} frames)")

# Finalizar
cap.release()
out.release()

print(f"\n✓ Vídeo gerado com sucesso!")
print(f"✓ Arquivo salvo em: {output_path}")
print(f"✓ Total de frames processados: {frames_processados}")
print(f"✓ Duração: {frames_processados / OUTPUT_FPS:.1f} segundos")