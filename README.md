# 📷 Detecção de Pessoas em Tempo Real com YOLOv8, OpenCV e Supervision

Este projeto realiza **detecção de pessoas em tempo real** utilizando o modelo YOLOv8 da Ultralytics, com captura de vídeo via webcam e visualização com caixas e rótulos usando a biblioteca **Supervision**. 

---

## ✨ Funcionalidades

- 📦 Detecção de pessoas com o modelo YOLOv8 pré-treinado (`yolov8n.pt`)
- 🧠 Supressão de avisos de log indesejados durante a importação do YOLO
- 🔲 Anotação visual com caixas e rótulos no vídeo
- 🖥️ Visualização em tempo real com OpenCV
- 📊 Impressão periódica do número de pessoas detectadas

---

## 📁 Requisitos

Instale as bibliotecas necessárias:

```bash
pip install opencv-python ultralytics supervision

