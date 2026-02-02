#!/usr/bin/env python3
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import time
import math
import cv2                                                                       # type: ignore
import numpy as np                                                               # type: ignore
import rclpy                                                                     # type: ignore
from rclpy.node import Node                                                      # type: ignore
from sensor_msgs.msg import Image, LaserScan                                     # type: ignore
from nav_msgs.msg import Odometry                                                # type: ignore
from geometry_msgs.msg import Twist                                              # type: ignore
from cv_bridge import CvBridge                                                   # type: ignore
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy         # type: ignore
import cv2
import random


class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__("limo_yolo")

        # PARAMETRI ROS
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("turn_speed_max", 1.5)           # Velocità angolare massima

        self.img_topic = self.get_parameter("image_topic").value
        self.turn_speed_max = self.get_parameter("turn_speed_max").value

        # CAMERA E IMMAGINE
        self.image_w = None         # Ampiezza immagine (width)

        self.bridge = CvBridge()

        # MODELLO
        self.model = None

        # CONTROLLO MOVIMENTO
        self.rate_hz = 20
        self.forward_speed = 0.4        # Velocità lineare
        self.align_tol = 0.05           # Errore tollerato in fase di riallineamento

        # VARIABILE DI STATO
        self.state = "FORWARD"  # Indica lo stato interno della FSM

        # ODOMETRIA
        # Odometria corrente
        self.x = None
        self.y = None
        self.yaw = None

        # Odometria di partenza
        self.starting_x = None
        self.starting_y = None
        self.starting_yaw = None

        # SEARCH E SCAN
        self.search_straight_distance = 1 # Distanza in rettilineo da percorrere in fase "FORWARD"

        # Odometria di partenza per la fase di ricerca
        self.search_start_x = None
        self.search_start_y = None
        self.search_start_yaw = None

        # Variabili di stato per la fase di "SCAN"
        self.turning_right = True
        self.turning_left = False
        self.realigning = False

        self.right_target_yaw = None
        self.left_target_yaw = None
        self.realigning_target_yaw = None

        # Variabile di stato che indica se una pianta malata è stata identificata
        self.ill_plant_detected = False
        self.ill_plant_detected_left = False
        self.ill_plant_detected_right = False

        # RILEVAMENTO OSTACOLI
        self.ranges = []

        self.obstacle_detected = False
        self.avoiding_dir = 1   # Sinistra di default
        self.avoiding = False
        self.aligning = False

        self.obstacle_threshold = 0.4   # Distanza di sicurezza a cui evitare un ostacolo
        self.current_distance = np.inf  # Distanza corrente da eventuali ostacoli

        # Odometria di partenza in fase di evitamento ostacolo
        self.start_avoiding_yaw = None

        # Logica di stop
        self.stopped = False

        # Variabili relative ai checkpoint (bivi)
        self.current_checkpoint = None

        self.rotated_at_checkpoint = False
        self.frontal_image = None
        self.lateral_image = None
        self.count_frontal_image = None
        self.count_lateral_image = None

        self.checkpoint_initial_yaw = None

        self.checkpoint1_reached = False
        self.checkpoint1_finished = False

        self.checkpoint2_reached = False
        self.checkpoint2_finished = False

        self.checkpoint3_reached = False
        self.checkpoint3_finished = False

        self.checkpoint4_reached = False

        # Coordinate dei checkpoint
        self.checkpoint1_x = 0.650
        self.checkpoint1_y = -2.000
        self.checkpoint2_x = 0.650
        self.checkpoint2_y = 0.525
        self.checkpoint3_x = -1.825
        self.checkpoint3_y = 0.525

        # Ultimo checkpoint -> posizione finale
        self.checkpoint4_x = -1.825
        self.checkpoint4_y = -2.000

        # -----------------------------
        # ROS I/O
        # -----------------------------
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos)    # Riceve odometria
        self.sub_image = self.create_subscription(Image, self.img_topic, self.on_image, 10)     # Riceve le immagini della camera
        self.sub_scan = self.create_subscription(LaserScan, "/scan", self.on_scan, 10)          # Riceve il LIDAR
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)                             # Invia le velocità
        self.pub_image = self.create_publisher(Image, "/yolo/annotated_image", 10)              # Pubblica le immagini annotate con bounding box

        self.control_timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)           # Ciclo di controllo principale

        self.get_logger().info(f"Node loaded...")


    # -----------------------------
    # CALLBACK PER ODOMETRIA
    # -----------------------------
    def odom_callback(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        if self.starting_x is None:
            self.starting_x = self.x
            self.starting_y = self.y
            self.starting_yaw = self.yaw
            self.get_logger().info("Odom initialized")

    # -----------------------------
    # CALLBACK PER CAMERA
    # -----------------------------
    def on_image(self, msg: Image):
        return


    # -----------------------------
    # CALLBACK PER LIDAR
    # -----------------------------
    def on_scan(self, msg: LaserScan):
        """
        Callback per i dati LiDAR. Aggiorna la distanza minima frontale 
        considerando un settore centrale di 40 campioni.
        """

        self.ranges = np.array(msg.ranges)

        n = len(self.ranges)
        center = n // 2
        left = max(center - 20, 0)
        right = min(center + 20, n)
        if left < right:
            self.current_distance = np.nanmin(self.ranges[left:right])
        else:
            self.current_distance = np.nanmin(self.ranges)

    # -----------------------------
    # METODI DI CONTROLLO
    # -----------------------------
    def distance_from_point(self, x0: float, y0: float) -> float:
        """
        Calcola la distanza euclidea dalla posizione corrente (self.x, self.y)
        a un punto dato (x0, y0).
        """
        return math.hypot(self.x - x0, self.y - y0)
    
    def quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """
        Calcola l'imbardata (yaw) da un quaternione.
        """
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy*qy + qz*qz)
        return math.atan2(siny, cosy)

    def normalize_angle(self, angle: float) -> float:
        """
        Normalizza un angolo all'intervallo [-pi, pi).
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def angle_error(self, target_angle: float) -> float:
        """
        Calcola l'errore angolare normalizzato tra l'angolo target e lo yaw corrente.
        """
        return self.normalize_angle(target_angle - self.yaw)

    def calculate_target_yaw(self, goal_x: float, goal_y: float) -> float:
        """
        Calcola l'angolo di imbardata (yaw) necessario per puntare verso il goal.
        """
        return math.atan2(goal_y - self.y, goal_x - self.x)
    
    def publish_twist(self, linear=0.0, angular=0.0):
        """
        Pubblica un comando di velocità lineare e angolare sul topic /cmd_vel.

        Args:
            linear (float): Velocità lineare in X (m/s).
            angular (float): Velocità angolare in Z (rad/s).
        """
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def publish_stop(self):
        """Pubblica un comando di velocità nullo (arresto del robot)."""
        self.publish_twist(0.0, 0.0)

    def print_message(self, message):
        self.get_logger().info(message)
        

    # -----------------------------
    # RILEVAMENTO OSTACOLI
    # -----------------------------
    def detect_obstacle(self):
        """
        Verifica la presenza di ostacoli.

        Prioritizza l'evitamento: un ostacolo è rilevato se la distanza LiDAR
        frontale è inferiore alla soglia.
        """
 
        # Rileva un ostacolo generico se la distanza LiDAR frontale è troppo piccola.
        self.obstacle_detected = self.current_distance < self.obstacle_threshold


    def align_to_next_checkpoint(self):
        goal_x = None
        goal_y = None

        # Determina la posizione del checkpoint a cui allinearsi (il successivo rispetto a quello raggiunto)
        if self.current_checkpoint is None:
            goal_x = self.checkpoint1_x
            goal_y = self.checkpoint1_y
        elif self.current_checkpoint == 1:
            goal_x = self.checkpoint2_x
            goal_y = self.checkpoint2_y       
        elif self.current_checkpoint == 2:
            goal_x = self.checkpoint3_x
            goal_y = self.checkpoint3_y    
        elif self.current_checkpoint == 3:
            goal_x = self.checkpoint4_x
            goal_y = self.checkpoint4_y

        # Si allinea al prossimo checkpoint
        target_yaw = self.calculate_target_yaw(goal_x, goal_y)
        angle_err = self.angle_error(target_yaw)
            
        if abs(angle_err) > self.align_tol:
            # Ruota con controllo proporzionale
            angular_vel = max(min(0.8 * angle_err, self.turn_speed_max), -self.turn_speed_max)
            self.publish_twist(0.0, angular_vel)
        else:
            self.aligning = False

            self.state = "FORWARD"

        return    

    def check_for_checkpoint_reached(self):

        # CHECKPOINT_1
        if not self.checkpoint1_reached and not self.checkpoint1_finished:
            d = self.distance_from_point(self.checkpoint1_x, self.checkpoint1_y)
            #print(d)
            if d <= 0.1:
                self.checkpoint1_reached = True
                self.checkpoint1_finished = True
                self.current_checkpoint = 1
                return 
            
        # CHECKPOINT_2
        if not self.checkpoint2_reached and not self.checkpoint2_finished:
            d = self.distance_from_point(self.checkpoint2_x, self.checkpoint2_y)
            #print(d)
            if d <= 0.1:
                self.checkpoint2_reached = True
                self.checkpoint2_finished = True
                self.current_checkpoint += 1
                return 
            
        # CHECKPOINT_3
        if not self.checkpoint3_reached and not self.checkpoint3_finished:
            d = self.distance_from_point(self.checkpoint3_x, self.checkpoint3_y)
            #print(d)
            if d <= 0.1:
                self.checkpoint3_reached = True
                self.checkpoint3_finished = True
                self.current_checkpoint += 1
                return      
            
        # CHECKPOINT_4
        if not self.checkpoint4_reached:
            d = self.distance_from_point(self.checkpoint4_x, self.checkpoint4_y)
            #print(d)
            if d <= 0.1:
                self.stopped = True     # Ferma definitivamente il robot

                self.publish_stop()
                return 


    # -----------------------------
    # METODI DI STATO
    # -----------------------------
    def avoid(self):
        # 1. Initialize Avoidance
        if self.start_avoiding_yaw is None:
            self.start_avoiding_yaw = self.yaw
            self.avoid_step = 0 # 0: Look Left, 1: Look Right, 2: Look Back
            self.avoid_target_yaw = None
            return

        # 2. Define the target angle for the current step
        if self.avoid_target_yaw is None:
            if self.avoid_step == 0:
                target_offset = math.pi / 2   # +90° (Left)
            elif self.avoid_step == 1:
                target_offset = -math.pi / 2  # -90° (Right)
            else:
                target_offset = -math.pi      # -180° (Back)
            
            self.avoid_target_yaw = self.normalize_angle(self.start_avoiding_yaw + target_offset)

        # 3. Rotate toward the target
        angle_error = self.angle_error(self.avoid_target_yaw)

        if abs(angle_error) > 0.1:
            direction = 1.0 if angle_error > 0 else -1.0
            self.publish_twist(0.0, direction * 0.5) # Slightly faster rotation
            return

        # 4. Rotation reached! Now check the path
        self.publish_stop()
        
        if not self.obstacle_detected:
            # PATH CLEAR: Exit avoidance and go forward
            self.start_avoiding_yaw = None
            self.avoid_target_yaw = None

            self.state = "ALIGN"
            return
        else:
            # PATH BLOCKED: Try the next direction
            self.avoid_step += 1
            self.avoid_target_yaw = None
            
            # If we've finished looking back, we just stop or force a move
            if self.avoid_step > 2:
                self.stopped = True

    # -----------------------------
    # CONTROL LOOP
    # -----------------------------
    def control_loop(self):
        """
        Ciclo di controllo principale del robot (FSM).

        Gestisce le transizioni di stato basandosi sulla posizione (checkpoint), 
        sulla sicurezza (evitamento ostacoli) e sulla logica di missione. Coordina 
        l'esecuzione dei comportamenti di navigazione, scansione e analisi.
        """

        if self.x is None or self.yaw is None or self.ranges is None:
            self.publish_stop()
            return

        self.check_for_checkpoint_reached()

        self.detect_obstacle()
        if self.obstacle_detected and self.state == "FORWARD":
            self.state = "AVOID"


        if self.state == "AVOID":
            self.avoid()
        elif self.state == "FORWARD":
            self.publish_twist(0.5, 0.0)
        elif self.state == "ALIGN":
            self.align_to_next_checkpoint()

        

# -----------------------------
# MAIN
# -----------------------------
def main():
    rclpy.init()
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()