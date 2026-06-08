import json
import bisect
import matplotlib.pyplot as plt
import numpy as np

class DataReader:
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.fps              = self.data.get('fps', 30.0404)
        self.diametro_tubo_mm = self.data.get('diametro_tubo_mm', 88.9)
        self.diametros        = sorted(self.data.get('diametros', []))
        self.odometria        = self.data.get('odometria', [])


        if 'frame_inicial' in self.data:
            self.frame_inicial = int(self.data['frame_inicial'])
        elif 'historico_pontos' in self.data and self.data['historico_pontos']:
            self.frame_inicial = min(int(k) for k in self.data['historico_pontos'].keys())
        else:
            self.frame_inicial = 0

        # Tempo absoluto do vídeo (s) correspondente a t_rel = 0
        self.t_offset_s = self.frame_inicial / self.fps

        self.vetor_tempo        = []
        self.vetor_deslocamento = []
        self._pre_computar_odometria()

    def get_mm_per_px(self, t_rel_ms: float) -> float:
        """Retorna mm/px para o instante t_rel_ms (ms relativos ao início do tracking)."""
        if not self.diametros:
            return 1.0
        idx = bisect.bisect_right([d[0] for d in self.diametros], t_rel_ms) - 1
        if idx < 0:
            idx = 0
        return self.diametro_tubo_mm / self.diametros[idx][1]

    def _pre_computar_odometria(self):
        dist_acumulada_mm = 0.0

        for i, (t_ms, px_atual) in enumerate(self.odometria):
            # Tempo absoluto do vídeo: offset + tempo relativo do tracking
            t_abs_s = self.t_offset_s + t_ms / 1000.0

            if i > 0:
                _, px_ant          = self.odometria[i - 1]
                delta_px           = px_atual - px_ant
                escala             = self.get_mm_per_px(t_ms)  # t_rel_ms
                dist_acumulada_mm += delta_px * escala

            self.vetor_tempo.append(t_abs_s)
            self.vetor_deslocamento.append(dist_acumulada_mm)

    def get_deslocamento_no_tempo(self, tempo_s: float) -> float:
        """Retorna deslocamento em mm para o tempo absoluto do vídeo em segundos."""
        idx = bisect.bisect_left(self.vetor_tempo, tempo_s)
        if idx >= len(self.vetor_deslocamento):
            return self.vetor_deslocamento[-1]
        return self.vetor_deslocamento[idx]


if __name__ == "__main__":
    reader = DataReader("CORTES-CENPES/PONTO 17/ELMLL25-228-OS006000765640-2025-12-15-T-09-34-00-D003.mp4_odometria_OF_sem_zoom.json")

    v_t = reader.vetor_tempo
    v_d = reader.vetor_deslocamento

    print(f"frame_inicial : {reader.frame_inicial}")
    print(f"t_offset_s    : {reader.t_offset_s:.3f} s")
    print(f"vetor_tempo   : {v_t[0]:.3f}s .. {v_t[-1]:.3f}s  ({len(v_t)} pts)")

    v_d_array = np.array(v_d)
    v_t_array = np.array(v_t)

    delta_d    = np.diff(v_d_array)
    delta_t    = np.diff(v_t_array)
    velocidade = delta_d / delta_t

    plt.plot(v_t_array[1:], velocidade, color='green', label='Velocidade (mm/s)')
    plt.xlabel('Tempo absoluto do vídeo (s)')
    plt.ylabel('Velocidade (mm/s)')
    plt.legend()
    plt.show()
