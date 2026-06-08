import open3d as o3d
import os

MODEL_PATH = "modelov2.obj"
OUTPUT_FILE = "pontos_selecionados.txt"

# 1. Carregar a malha
if not os.path.exists(MODEL_PATH):
    print(f"Erro: Arquivo '{MODEL_PATH}' não encontrado.")
    exit()

mesh = o3d.io.read_triangle_mesh(MODEL_PATH)
mesh.compute_vertex_normals()

# 2. Configurar o Seletor Moderno
vis = o3d.visualization.VisualizerWithVertexSelection()
vis.create_window(window_name="Seletor Profissional - Shift+Clique", width=1200, height=800)
vis.add_geometry(mesh)

print("\n--- COMANDOS DE ATALHO ---")
print("[SHIFT + Clique Esq.] Selecionar Vértice")
print("[W] Ativar/Desativar Aramado (Wireframe)")
print("[L] Alternar Iluminação")
print("[R] Resetar Câmera")
print("[Q] Sair e Salvar")
print("--------------------------\n")

vis.run()
vis.destroy_window()

# 3. Processar e Salvar os Pontos
picked_points = vis.get_picked_points()

if not picked_points:
    print("Nenhum ponto foi selecionado. O arquivo não foi criado.")
else:
    # Criar lista de coordenadas formatada
    coords = [p.coord for p in picked_points]
    indices = [p.index for p in picked_points]

    # Salvar em TXT (compatível com Excel, MATLAB, Python, etc.)
    with open(OUTPUT_FILE, "w") as f:
        f.write("Index, X, Y, Z\n")
        for i, (idx, c) in enumerate(zip(indices, coords)):
            f.write(f"{idx}, {c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}\n")

    print(f"\nSucesso! {len(coords)} pontos salvos em: {OUTPUT_FILE}")
    print("Exemplo do primeiro ponto:")
    print(f"ID: {indices[0]} -> {coords[0]}")

