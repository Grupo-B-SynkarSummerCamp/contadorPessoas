# ------------------------- IMPORTAÇÕES ------------------------- #

import os
import sys
import time
import cv2
import supervision as sv  # Biblioteca para anotações visuais (caixas, rótulos)

# ------------------ FUNÇÕES AUXILIARES PARA STDERR ------------------ #

# Suprime mensagens de erro no stderr temporariamente (útil para evitar avisos na importação do YOLO)
def suppress_stderr():
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    return saved_stderr

def restore_stderr(saved_stderr):
    sys.stderr.flush()
    os.dup2(saved_stderr, sys.stderr.fileno())
    os.close(saved_stderr)

# Define nível de log do PyTorch para evitar mensagens excessivas
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

# Suprime stderr antes de importar o YOLO para evitar mensagens de aviso
saved_stderr = suppress_stderr()
from ultralytics import YOLO  # Importa modelo de detecção de objetos
restore_stderr(saved_stderr)

# ----------------------- INICIALIZAÇÕES ------------------------ #

# Carrega modelo YOLO pré-treinado (versão leve)
model = YOLO('yolov8n.pt')

# Inicializa anotadores visuais (caixa e rótulo)
box_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)

# Captura de vídeo pela webcam
cap = cv2.VideoCapture(0) # O número do parâmetro é a entrada USB da câmera
cv2.namedWindow('Detecção em Tempo Real', cv2.WINDOW_NORMAL)

# Para controlar intervalo entre prints no terminal
last_print_time = time.time()

# --------------------- LOOP PRINCIPAL ---------------------- #

while True:
    ret, frame = cap.read()  # Lê frame da webcam

    if not ret:
        print("Falha ao capturar frame, encerrando...")
        break

    # Executa a detecção com YOLO (sem mostrar progresso - verbose=False)
    results = model(frame, imgsz=640, verbose=False)[0]

    # Converte os resultados para o formato da biblioteca Supervision
    detections = sv.Detections.from_ultralytics(results)

    # Filtra apenas detecções da classe 'pessoa' (class_id == 0)
    detections = detections[detections.class_id == 0]

    # Gera rótulos formatados para exibição no vídeo
    labels = [
        f"{model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _, _ in detections
    ]

    # Adiciona caixas delimitadoras e rótulos no frame
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # Exibe o frame anotado na janela
    cv2.imshow('Detecção em Tempo Real', frame)

    # A cada 5 segundos, imprime o número de pessoas detectadas
    if time.time() - last_print_time >= 5:
        print(f"Pessoas detectadas: {len(detections)}")
        last_print_time = time.time()

    # Encerra o loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------- FINALIZAÇÃO ----------------------- #

cap.release()              # Libera a câmera
cv2.destroyAllWindows()    # Fecha janelas abertas
