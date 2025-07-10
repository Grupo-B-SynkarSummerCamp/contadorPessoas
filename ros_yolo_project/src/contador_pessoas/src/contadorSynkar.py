#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ------------------------- IMPORTAÇÕES ------------------------- #
import rospy
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

# ----------------------- CLASSE DO NÓ ROS ------------------------ #

class PeopleCounter:
    def __init__(self):
        # --- Inicialização do Nó ---
        rospy.init_node('people_counter_node', anonymous=True)
        rospy.loginfo("Nó de contagem de pessoas iniciado.")

        # --- Carregamento do Modelo YOLO ---
        # O caminho agora aponta para a pasta 'models' dentro do nosso pacote.
        # Supondo que o Dockerfile copie o pacote para /root/catkin_ws/src/
        model_path = rospy.get_param('~model_path', '/root/catkin_ws/src/contador_pessoas/models/yolov8n.pt')
        self.model = YOLO(model_path)
        rospy.loginfo(f"Modelo YOLO carregado de: {model_path}")

        # --- Inicializadores do Supervision e CvBridge ---
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)
        self.bridge = CvBridge()

        # --- Publishers (Publicadores) ---
        # Publica a imagem com as detecções para visualização (debug)
        self.image_pub = rospy.Publisher("/person_detector/image_annotated", Image, queue_size=1)
        # Publica o número de pessoas detectadas
        self.count_pub = rospy.Publisher("/person_detector/person_count", Int32, queue_size=1)

        # --- Subscriber (Assinante) ---
        # Assina o tópico de imagem da câmera do robô.
        # IMPORTANTE: Altere '/camera/image_raw' para o tópico real da câmera do seu robô.
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)

    def image_callback(self, msg):
        """
        Função chamada toda vez que uma nova imagem chega no tópico da câmera.
        """
        try:
            # Converte a mensagem de imagem do ROS para um frame do OpenCV
            cv_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # --- Lógica de Detecção (quase idêntica à sua original) ---
        results = self.model(cv_frame, imgsz=640, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Filtra apenas detecções da classe 'pessoa' (class_id == 0)
        detections = detections[detections.class_id == 0]

        # Gera rótulos
        labels = [
            f"{self.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _ in detections
        ]

        # Anota o frame com as caixas e rótulos
        annotated_frame = self.box_annotator.annotate(scene=cv_frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # --- Publicação dos Resultados ---
        
        # 1. Publica o número de pessoas
        person_count = len(detections)
        self.count_pub.publish(Int32(person_count))

        # 2. Publica a imagem anotada
        try:
            ros_image_annotated = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.image_pub.publish(ros_image_annotated)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def run(self):
        """
        Mantém o nó em execução.
        """
        rospy.spin()

# ----------------------- PONTO DE ENTRADA DO SCRIPT ------------------------ #

if __name__ == '__main__':
    try:
        counter = PeopleCounter()
        counter.run()
    except rospy.ROSInterruptException:
        pass