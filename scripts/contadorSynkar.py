#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
import rospkg
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import supervision as sv


class PersonCounterNode:
    def __init__(self):
        # Inicializa nó ROS
        rospy.init_node('person_counter', anonymous=False)

        # Obtém caminho padrão do modelo dentro do pacote 'contadorPessoas'
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('contadorPessoas')
        default_model = os.path.join(pkg_path, 'models', 'yolov8n.pt')

        # Parâmetro para caminho do modelo (pode ser sobrescrito via launch)
        model_path = rospy.get_param('~model_path', default_model)
        imgsz = rospy.get_param('~imgsz', 640)

        # Carrega modelo YOLO
        rospy.loginfo(f"Carregando YOLO a partir de {model_path}...")
        self.model = YOLO(model_path)
        rospy.loginfo("Modelo carregado com sucesso.")

        # Bridge para conversão ROS2cv
        self.bridge = CvBridge()

        # Publicador do número de pessoas
        self.pub_count = rospy.Publisher('/person_count', Int32, queue_size=1)

        # Anotadores opcionais
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)

        # Assina tópico de imagem
        self.sub_image = rospy.Subscriber(
            '/camera/color/image_raw', Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

        rospy.loginfo("Aguardando imagens em /camera/color/image_raw...")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"Erro ao converter imagem: {e}")
            return

        # Detecção YOLO
        results = self.model(frame, imgsz=self.model.args.imgsz or 640, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        person_dets = detections[detections.class_id == 0]
        count = len(person_dets)

        # Publica e log
        self.pub_count.publish(count)
        rospy.loginfo(f"Pessoas detectadas neste frame: {count}")

        # Para visualização (opcional)
        # labels = [f"{self.model.names[cid]} {conf:0.2f}" for *_, conf, cid, *_ in person_dets]
        # frame = self.box_annotator.annotate(scene=frame, detections=person_dets)
        # frame = self.label_annotator.annotate(scene=frame, detections=person_dets, labels=labels)
        # cv2.imshow("Detecção de Pessoas", frame)
        # cv2.waitKey(1)

    def run(self):
        rospy.spin()
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        node = PersonCounterNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
