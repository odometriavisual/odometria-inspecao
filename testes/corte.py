from moviepy import VideoFileClip

# 1. Carrega o vídeo
video = VideoFileClip("preview_pose.mp4")

# 2. Corta usando o novo nome do método: subclipped
video_cortado = video.subclipped(0, 60)

# 3. Salva o resultado
video_cortado.write_videofile("preview_1minuto.mp4")

# 4. Fecha o arquivo
video.close()