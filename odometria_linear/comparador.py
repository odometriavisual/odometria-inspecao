import cv2
import numpy as np
import json
from tkinter import Tk, filedialog, messagebox
import os

# ================= CONFIGURAÇÕES =================
GRAPH_HEIGHT = 300  # Altura do gráfico
MARGIN = 50  # Margem para legendas
OUTPUT_FPS = 30  # FPS do vídeo de saída


# ================= FUNÇÕES DE APOIO =================

def load_odometria(json_path, aplicar_compensacao_zoom=False, forcar_frame_inicial=None):
    """Carrega arquivo JSON de odometria com opção de compensação de zoom e sincronização forçada"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extrair dados de odometria [tempo_ms, distancia_px]
        odometria = data.get('odometria', [])
        fps = data.get('fps', 30.0)
        diametros = data.get('diametros', [])

        print(f"  Odometria: {len(odometria)} pontos")
        print(f"  Diâmetros: {len(diametros)} medições")

        if odometria and len(odometria) > 0:
            print(f"  Primeiro timestamp: {odometria[0][0]:.2f} ms")
            print(f"  Último timestamp: {odometria[-1][0]:.2f} ms")

        # Tentar encontrar frame inicial
        frame_inicial = 0

        # Se foi fornecido um frame inicial forçado, usar ele
        if forcar_frame_inicial is not None:
            frame_inicial = forcar_frame_inicial
            print(f"  Frame inicial (FORÇADO): {frame_inicial}")
        else:
            # Método 1: Ver se tem histórico de pontos (optical flow)
            if 'historico_pontos' in data and data['historico_pontos']:
                frames_hist = [int(k) for k in data['historico_pontos'].keys()]
                if frames_hist:
                    frame_inicial = min(frames_hist)
                    print(f"  Frame inicial (do histórico): {frame_inicial}")

            # Método 2: Verificar se há hora_inicio_ultrassom para calcular offset
            if frame_inicial == 0 and 'hora_inicio_ultrassom' in data and data['hora_inicio_ultrassom']:
                hora_video = data.get('hora_video', '')
                hora_inicio_us = data['hora_inicio_ultrassom']

                if hora_video and hora_inicio_us:
                    try:
                        # Converter strings de tempo para segundos
                        def time_to_seconds(time_str):
                            h, m, s = map(float, time_str.split(':'))
                            return h * 3600 + m * 60 + s

                        video_sec = time_to_seconds(hora_video)
                        us_sec = time_to_seconds(hora_inicio_us)

                        # Diferença em segundos
                        diff_sec = us_sec - video_sec
                        if diff_sec < 0:
                            diff_sec += 86400  # Adicionar 24h se passou da meia-noite

                        frame_inicial = int(diff_sec * fps)
                        print(f"  Frame inicial (do timestamp): {frame_inicial} (diff: {diff_sec:.2f}s)")
                    except Exception as e:
                        print(f"  Aviso: Não foi possível calcular offset do timestamp: {e}")

            # Método 3: Calcular a partir do primeiro tempo de odometria se ainda for 0
            if frame_inicial == 0 and odometria and len(odometria) > 0:
                primeiro_tempo_ms = odometria[0][0]
                frame_inicial = int((primeiro_tempo_ms / 1000.0) * fps)
                if frame_inicial > 0:
                    print(f"  Frame inicial (calculado do tempo): {frame_inicial}")

        # Processar diâmetros se disponíveis
        diametros_por_frame = {}
        diametro_referencia = None

        if diametros and len(diametros) > 0:
            for item in diametros:
                if len(item) >= 2:
                    frame_diam, diam_valor = item[0], item[1]
                    diametros_por_frame[frame_diam] = diam_valor
                    if diametro_referencia is None:
                        diametro_referencia = diam_valor

            print(f"  Diâmetro de referência: {diametro_referencia:.2f} px")
            if len(diametros_por_frame) > 1:
                diam_min = min(diametros_por_frame.values())
                diam_max = max(diametros_por_frame.values())
                print(
                    f"  Variação de diâmetro: {diam_min:.2f} - {diam_max:.2f} px ({((diam_max / diam_min) - 1) * 100:.1f}% variação)")

        # Converter tempo relativo para frame absoluto e aplicar compensação de zoom
        dados_frame = []
        distancia_acumulada_compensada = 0.0

        for i, (tempo_ms, dist_px) in enumerate(odometria):
            # tempo_ms é relativo ao início da odometria
            frame_relativo = int((tempo_ms / 1000.0) * fps)
            frame_absoluto = frame_inicial + frame_relativo

            if aplicar_compensacao_zoom and diametro_referencia and len(diametros_por_frame) > 0:
                # Encontrar o diâmetro mais próximo deste frame
                diametro_atual = diametro_referencia
                menor_diferenca = float('inf')

                for frame_diam, diam_valor in diametros_por_frame.items():
                    diff = abs(frame_absoluto - frame_diam)
                    if diff < menor_diferenca:
                        menor_diferenca = diff
                        diametro_atual = diam_valor

                # Calcular fator de compensação
                # Se o diâmetro aumentou (zoom in), os deslocamentos devem ser reduzidos
                # Se o diâmetro diminuiu (zoom out), os deslocamentos devem ser aumentados
                fator_zoom = diametro_referencia / diametro_atual

                # Aplicar compensação ao incremento de distância
                if i > 0:
                    dist_anterior = odometria[i - 1][1]
                    incremento = dist_px - dist_anterior
                    incremento_compensado = incremento * fator_zoom
                    distancia_acumulada_compensada += incremento_compensado
                else:
                    # Primeiro ponto
                    distancia_acumulada_compensada = dist_px * fator_zoom

                dados_frame.append([frame_absoluto, distancia_acumulada_compensada])
            else:
                # Sem compensação de zoom
                dados_frame.append([frame_absoluto, dist_px])

        if aplicar_compensacao_zoom and diametro_referencia:
            print(f"  ✓ Compensação de zoom aplicada (ref: {diametro_referencia:.2f} px)")
            if len(dados_frame) > 0:
                print(f"  Distância original final: {odometria[-1][1]:.2f} px")
                print(f"  Distância compensada final: {dados_frame[-1][1]:.2f} px")

        if len(dados_frame) > 0:
            print(f"  Range de frames: {dados_frame[0][0]} - {dados_frame[-1][0]}")

        return dados_frame, data, frame_inicial
    except Exception as e:
        print(f"Erro ao carregar {json_path}: {e}")
        import traceback
        traceback.print_exc()
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


def draw_graph(width, height, dados_list, current_frame, max_dist, frame_inicial, colors, names):
    """Desenha gráfico de comparação das odometrias (suporta até 3)"""
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

    # Encontrar frame máximo
    max_frame = frame_inicial
    for dados in dados_list:
        if dados and len(dados) > 0:
            max_frame = max(max_frame, dados[-1][0])

    if max_frame == frame_inicial:
        max_frame = frame_inicial + 1000

    # Grade vertical (tempo/frames)
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

    # Função para converter coordenadas
    def frame_to_x(frame):
        if frame_range == 0:
            return margin_left
        return margin_left + int(((frame - frame_inicial) / frame_range) * graph_w)

    def dist_to_y(dist):
        return margin_top + int((1 - dist / max_dist) * graph_h)

    # Desenhar linhas de cada odometria
    for idx, dados in enumerate(dados_list):
        if dados and len(dados) > 1:
            pts = []
            for frame, dist in dados:
                x = frame_to_x(frame)
                y = dist_to_y(dist)
                pts.append([x, y])

            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(graph, [pts], False, colors[idx], 2)

    # Linha vertical do frame atual
    current_x = frame_to_x(current_frame)
    cv2.line(graph, (current_x, margin_top), (current_x, margin_top + graph_h),
             (255, 0, 0), 2)

    # Círculos nos pontos atuais
    for idx, dados in enumerate(dados_list):
        if dados:
            dist = interpolate_distance(dados, current_frame)
            cv2.circle(graph, (current_x, dist_to_y(dist)), 6, colors[idx], -1)
            darker_color = tuple(int(c * 0.7) for c in colors[idx])
            cv2.circle(graph, (current_x, dist_to_y(dist)), 8, darker_color, 2)

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


def draw_info_panel(width, distancias, frame, names, colors):
    """Desenha painel de informações (suporta até 3 odometrias)"""
    num_odometrias = len(distancias)
    panel_height = 120 if num_odometrias > 2 else 100
    panel = np.ones((panel_height, width, 3), dtype=np.uint8) * 40

    # Desenhar cada odometria
    y_offset = 10
    for idx, (dist, name, color) in enumerate(zip(distancias, names, colors)):
        label = chr(65 + idx)  # A, B, C

        cv2.rectangle(panel, (10, y_offset), (30, y_offset + 20), color, -1)
        cv2.putText(panel, f"{label}: {name}",
                    (40, y_offset + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, f"{dist:.1f} px",
                    (40, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    tuple(min(255, int(c * 1.3)) for c in color), 2)

        y_offset += 40


    return panel


# ================= EXECUÇÃO PRINCIPAL =================
print("=== GERADOR DE VÍDEO COMPARATIVO (Até 3 Odometrias) ===\n")

# Seleção de arquivos
Tk().withdraw()

print("Selecione o arquivo de Odometria A (VERDE):")
path_odom_a = filedialog.askopenfilename(title="Odometria A", filetypes=[("JSON", "*.json")])
if not path_odom_a:
    print("Nenhum arquivo selecionado.")
    exit()

print("\nSelecione o arquivo de Odometria B (AZUL):")
path_odom_b = filedialog.askopenfilename(title="Odometria B", filetypes=[("JSON", "*.json")])
if not path_odom_b:
    print("Nenhum arquivo selecionado.")
    exit()

# Perguntar se quer adicionar terceira odometria
adicionar_terceira = messagebox.askyesno(
    "Terceira Odometria",
    "Deseja adicionar uma terceira odometria (C - VERMELHA) para comparação?\n\n" +
    "Será aplicada compensação de zoom baseada nos diâmetros do tubo."
)

path_odom_c = None
if adicionar_terceira:
    print("\nSelecione o arquivo de Odometria C (VERMELHA) - com compensação de zoom:")
    path_odom_c = filedialog.askopenfilename(title="Odometria C (com compensação)",
                                             filetypes=[("JSON", "*.json")])
    if not path_odom_c:
        print("Nenhum arquivo selecionado para C, continuando apenas com A e B.")

print("\nSelecione o arquivo de vídeo:")
video_path = filedialog.askopenfilename(title="Vídeo", filetypes=[("Video", "*.mp4 *.avi *.mov")])
if not video_path:
    print("Nenhum arquivo selecionado.")
    exit()

# Perguntar onde salvar o vídeo de saída
print("\nEscolha onde salvar o vídeo comparativo:")
output_path = filedialog.asksaveasfilename(
    title="Salvar vídeo como",
    defaultextension=".mp4",
    filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")]
)
if not output_path:
    print("Nenhum destino selecionado.")
    exit()

# Carregar odometrias A e B primeiro (sem forçar frame inicial)
print("\n" + "=" * 60)
print("Carregando Odometria A (sem compensação)...")
dados_a, info_a, frame_inicial_a = load_odometria(path_odom_a, aplicar_compensacao_zoom=False)

print("\n" + "=" * 60)
print("Carregando Odometria B (sem compensação)...")
dados_b, info_b, frame_inicial_b = load_odometria(path_odom_b, aplicar_compensacao_zoom=False)

if dados_a is None or dados_b is None:
    print("Erro ao carregar arquivos de odometria A ou B.")
    exit()

# Usar o menor frame inicial de A e B como referência
frame_inicial_referencia = min(frame_inicial_a, frame_inicial_b)
print(f"\nFrame inicial de referência (A e B): {frame_inicial_referencia}")

# Agora carregar C forçando o mesmo frame inicial
dados_c = None
frame_inicial_c = frame_inicial_referencia
if path_odom_c:
    print("\n" + "=" * 60)
    print("Carregando Odometria C (COM compensação de zoom)...")
    print(f"Forçando sincronização no frame {frame_inicial_referencia}...")
    dados_c, info_c, _ = load_odometria(path_odom_c,
                                        aplicar_compensacao_zoom=True,
                                        forcar_frame_inicial=frame_inicial_referencia)

print("\n" + "=" * 60)

# Preparar listas para processamento
dados_list = [dados_a, dados_b]
frame_iniciais = [frame_inicial_a, frame_inicial_b]
paths = [path_odom_a, path_odom_b]
colors = [(0, 200, 0), (200, 0, 0)]  # Verde, Azul
names = []

if dados_c is not None:
    dados_list.append(dados_c)
    frame_iniciais.append(frame_inicial_c)
    paths.append(path_odom_c)
    colors.append((0, 0, 200))  # Vermelho

# Criar nomes curtos
for path in paths:
    name = os.path.basename(path).replace("_odometria_OF.json", "").replace("_odometria.json", "")[-25:]
    names.append(name)

# Usar o menor frame inicial como ponto de partida
frame_inicial = min(frame_iniciais)

print("\n" + "=" * 60)
print("RESUMO:")
for idx, (dados, name, inicial) in enumerate(zip(dados_list, names, frame_iniciais)):
    label = chr(65 + idx)
    if dados and len(dados) > 0:
        print(f"  {label}: {len(dados)} pontos (início: frame {inicial}, range: {dados[0][0]}-{dados[-1][0]}) - {name}")
    else:
        print(f"  {label}: ERRO ao carregar")
print(f"\n  Frame inicial de comparação: {frame_inicial}")
print("=" * 60)

# Abrir vídeo
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Erro ao abrir vídeo.")
    exit()

h_orig, w_orig = frame.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\n✓ Vídeo: {w_orig}x{h_orig} @ {fps:.1f} FPS ({total_frames} frames)")

# Usar resolução original do vídeo
video_w = w_orig
video_h = h_orig

# Encontrar distância máxima para escala do gráfico
max_dist = 0
for dados in dados_list:
    if dados:
        max_dist = max(max_dist, max([d[1] for d in dados]))
max_dist *= 1.1  # 10% de margem

print(f"Distância máxima: {max_dist:.1f} px")

# Configurar VideoWriter
panel_height = 120 if len(dados_list) > 2 else 100
output_w = video_w
output_h = video_h + GRAPH_HEIGHT + panel_height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (output_w, output_h))

if not out.isOpened():
    print("Erro ao criar arquivo de vídeo.")
    exit()

print(f"\nGerando vídeo: {output_w}x{output_h} @ {OUTPUT_FPS} FPS")
print(f"Processando do frame {frame_inicial} ao {total_frames}...")
print("=" * 60)

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
    distancias = [interpolate_distance(dados, current_frame) for dados in dados_list]

    # Desenhar informações no vídeo
    overlay_height = 70 if len(dados_list) <= 2 else 90
    cv2.rectangle(frame, (0, 0), (video_w, overlay_height), (0, 0, 0), -1)

    cv2.putText(frame, f"Frame: {current_frame}/{total_frames}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mostrar valores de cada odometria
    x_pos = 10
    for idx, (dist, name, color) in enumerate(zip(distancias, names, colors)):
        label = chr(65 + idx)
        text_color = tuple(min(255, int(c * 1.3)) for c in color)
        cv2.putText(frame, f"{label}: {dist:.1f}px",
                    (x_pos, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        x_pos += 150

    # Desenhar gráfico
    graph = draw_graph(video_w, GRAPH_HEIGHT, dados_list, current_frame,
                       max_dist, frame_inicial, colors, names)

    # Desenhar painel de informações
    info_panel = draw_info_panel(video_w, distancias, current_frame, names, colors)

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

print("\n" + "=" * 60)
print("✓ Vídeo gerado com sucesso!")
print(f"✓ Arquivo salvo em: {output_path}")
print(f"✓ Total de frames processados: {frames_processados}")
print(f"✓ Duração: {frames_processados / OUTPUT_FPS:.1f} segundos")
print("=" * 60)