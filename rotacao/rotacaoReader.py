import json
import numpy as np
import os


class RotacaoReader:
    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            dados = json.load(f)

        dados.sort(key=lambda d: d["frame"])

        self.vetor_tempo  = [d["timestamp_s"] for d in dados]
        self.vetor_angulo = [d["angulo_graus"] for d in dados]
        self.vetor_frames = [d["frame"]        for d in dados]
        self.eixo         = dados[0]["eixo"]
        self.n_capturas   = len(dados)

    def resumo(self):
        print(f"RotacaoReader — {self.n_capturas} capturas | eixo={self.eixo}")
        for i, (t, a, f) in enumerate(zip(self.vetor_tempo, self.vetor_angulo, self.vetor_frames)):
            print(f"  [{i}] frame={f:6d}  t={t:8.3f}s  angulo={a:+.2f} graus")


if __name__ == "__main__":
    import sys
    raiz = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    path = r"CORTES-CENPES\PONTO 12\ELMLL25-228-OS006000765640-2025-12-15-T-07-40-35-D009_rotacao.json"
    r = RotacaoReader(os.path.join(raiz, path))
    r.resumo()
