import json
import os
import sys
import tkinter as tk
from tkinter import filedialog
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from typing import List


# =============================================================================
#  CONFIGURAÇÕES — copie exatamente do script de preview
# =============================================================================

# Qual componente do rvec bruto contém a rotação que você quer
# 0 = X  |  1 = Y  |  2 = Z
EIXO_FONTE = 1

# Sentido: 1.0 = mesmo sentido  |  -1.0 = inverte
ESCALA = -1.0

# Eixo de destino (só salvo nos metadados, não afeta o cálculo)
# 0 = X (vermelho)  |  1 = Y (verde)  |  2 = Z (azul)
EIXO_DESTINO = 1

# =============================================================================
#  FIM DAS CONFIGURAÇÕES
# =============================================================================


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


def carregar_poses(caminho: str) -> List[dict]:
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    dados = [d for d in dados if d.get("sucesso", False)]
    dados.sort(key=lambda d: d["timestamp_s"])
    print(f"[OK] {len(dados)} poses carregadas  "
          f"({dados[0]['timestamp_s']:.3f}s – {dados[-1]['timestamp_s']:.3f}s)")
    return dados


def extrair_angulo_absoluto(dados: List[dict]) -> List[list]:
    """
    Para cada pose, pega o rvec bruto, extrai só a componente EIXO_FONTE,
    aplica ESCALA, converte para graus e retorna [timestamp_ms, angulo_graus].
    """
    nomes = ["X", "Y", "Z"]
    print(f"  eixo_fonte = {EIXO_FONTE} ({nomes[EIXO_FONTE]})  |  escala = {ESCALA}")

    odometria = []
    for d in dados:
        # rvec vem como quaternion no JSON — converte para rotvec
        quat  = d["quaternion"]                          # [x, y, z, w]
        rvec  = Rotation.from_quat(quat).as_rotvec()    # [rx, ry, rz] rad

        angulo_rad  = float(rvec[EIXO_FONTE]) * ESCALA
        angulo_grau = float(np.degrees(angulo_rad))

        timestamp_ms = int(round(d["timestamp_s"] * 1000))
        odometria.append([timestamp_ms, round(angulo_grau, 4)])

    return odometria


def inferir_metadados_do_nome(caminho_json: str) -> dict:
    """
    Tenta extrair data, hora e nome do vídeo a partir do nome do arquivo
    pose_dados.json ou do diretório pai — preenchimento melhor que vazio.
    """
    base = os.path.basename(os.path.dirname(caminho_json))
    # tenta pegar do diretório pai (geralmente tem o nome do vídeo)
    video  = base if base else os.path.splitext(os.path.basename(caminho_json))[0]
    data   = ""
    hora   = ""

    # padrão comum: ...-YYYY-MM-DD-T-HH-MM-SS-...
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})-T-(\d{2}-\d{2}-\d{2})", video)
    if m:
        data = m.group(1)
        hora = m.group(2).replace("-", ":")

    return {"video": video, "data_video": data, "hora_video": hora}


def salvar_resultado(odometria: List[list],
                     caminho_entrada: str,
                     caminho_saida: str):
    meta = inferir_metadados_do_nome(caminho_entrada)
    nomes_eixo = {0: "X", 1: "Y", 2: "Z"}

    saida = {
        "video":        meta["video"],
        "data_video":   meta["data_video"],
        "hora_video":   meta["hora_video"],
        "fps":          30.0,
        "metodologia":  "rotacao_pose",
        "eixo_fonte":   EIXO_FONTE,
        "eixo_fonte_nome": nomes_eixo[EIXO_FONTE],
        "eixo_destino": EIXO_DESTINO,
        "eixo_destino_nome": nomes_eixo[EIXO_DESTINO],
        "escala":       ESCALA,
        "unidade":      "graus",
        "odometria":    odometria,
    }

    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)

    print(f"[OK] {len(odometria)} entradas salvas em:")
    print(f"     {caminho_saida}")

    # resumo rápido
    angulos = [e[1] for e in odometria]
    print(f"     min={min(angulos):.2f}°  max={max(angulos):.2f}°  "
          f"amplitude={max(angulos)-min(angulos):.2f}°")


if __name__ == "__main__":
    print("=== Extrator de Odometria de Rotação ===")
    print(f"    eixo_fonte = {EIXO_FONTE}  |  escala = {ESCALA}  |  eixo_destino = {EIXO_DESTINO}\n")

    aqui = os.path.dirname(os.path.abspath(__file__))

    # ── pose_dados.json ───────────────────────────────────────────────────────
    json_padrao = os.path.join(aqui, "pose_dados.json")
    if os.path.exists(json_padrao):
        caminho_json = json_padrao
        print(f"[AUTO] {caminho_json}")
    else:
        caminho_json = pedir_arquivo("Selecione pose_dados.json", [("JSON", "*.json")])
    if not caminho_json:
        sys.exit("Cancelado.")

    dados = carregar_poses(caminho_json)
    if len(dados) < 2:
        sys.exit("[ERRO] Mínimo 2 poses necessárias.")

    # ── extrai ângulos ────────────────────────────────────────────────────────
    odometria = extrair_angulo_absoluto(dados)

    # ── caminho de saída ──────────────────────────────────────────────────────
    nome_saida_padrao = os.path.join(
        os.path.dirname(caminho_json),
        "odometria_rotacao.json"
    )
    resp = input(f"\nArquivo de saída [{nome_saida_padrao}]: ").strip()
    caminho_saida = resp if resp else nome_saida_padrao

    salvar_resultado(odometria, caminho_json, caminho_saida)
    print("\n[CONCLUÍDO]")