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
import joblib
import random


class FiniteStateMachine(Node):
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
        self.checkpoint4_finished = False

        self.checkpoint5_reached = False
        self.checkpoint5_finished = False

        self.checkpoint6_reached = False

        # Coordinate dei checkpoint
        self.checkpoint1_x = 0.525
        self.checkpoint1_y = -2.000
        self.checkpoint2_x = 0.525
        self.checkpoint2_y = 0.525
        self.checkpoint3_x = 0.525
        self.checkpoint3_y = 1.875
        self.checkpoint4_x = -1.975
        self.checkpoint4_y = 1.875
        self.checkpoint5_x = -1.975
        self.checkpoint5_y = 0.500

        # Ultimo checkpoint -> posizione finale
        self.checkpoint6_x = -1.975
        self.checkpoint6_y = -2.000

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

        self.load_model()
        self.get_logger().info(f"Node loaded...")

    # -----------------------------
    # CARICAMENTO MODELLO SVM
    # -----------------------------
    def load_model(self):
        """
        Carica il modello SVM e le etichette delle classi dai file joblib.

        Inizializza:
            self.model: Il classificatore caricato.
            self.class_names: Le etichette associate al modello.
        """
        


        # Espande il simbolo '~' o recupera automaticamente /home/nome_utente
        home = os.path.expanduser("~")
        model_path = os.path.join(home, "Agritech/workspace/assets/leaf_svm_model.joblib")
        labels_path = os.path.join(home, "Agritech/workspace/assets/leaf_labels.joblib")

        t0 = time.time()
        self.model = joblib.load(model_path)
        self.class_names = joblib.load(labels_path)
        self.get_logger().info(f"Loaded SVM model in {time.time() - t0:.2f}s")

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
        """
        Callback per l'elaborazione delle immagini.

        Converte il messaggio ROS in OpenCV, gestisce il salvataggio dei frame ai 
        checkpoint e analizza la presenza di pixel "verde chiaro" per il 
        rilevamento di piante potenzialmente malate.

        Args:
            msg (Image): Messaggio immagine ricevuto dal sensore.
        """

        # Converte da ROS a OpenCV
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return
        
        # Salva l'immagine per CHECKPOINT_1
        if self.checkpoint1_reached and self.frontal_image is None and not self.checkpoint1_finished:
            self.frontal_image = img_bgr.copy()
            self.count_frontal_image = self.count_ill_pixels(self.frontal_image)
            return

        if self.rotated_at_checkpoint and self.lateral_image is None and not self.checkpoint1_finished:
            self.lateral_image = img_bgr.copy()
            self.count_lateral_image = self.count_ill_pixels(self.lateral_image)
            return
        
        # Salva l'immagine per CHECKPOINT_2
        if self.checkpoint2_reached and self.frontal_image is None and not self.checkpoint2_finished:
            self.frontal_image = img_bgr.copy()
            self.count_frontal_image = self.count_ill_pixels(self.frontal_image)
            return

        if self.rotated_at_checkpoint and self.lateral_image is None and not self.checkpoint2_finished:
            self.lateral_image = img_bgr.copy()
            self.count_lateral_image = self.count_ill_pixels(self.lateral_image)
            return
        
        # Salva l'immagine per CHECKPOINT_3
        if self.checkpoint3_reached and self.frontal_image is None and not self.checkpoint3_finished:
            self.frontal_image = img_bgr.copy()
            self.count_frontal_image = self.count_ill_pixels(self.frontal_image)
            return

        if self.rotated_at_checkpoint and self.lateral_image is None and not self.checkpoint3_finished:
            self.lateral_image = img_bgr.copy()
            self.count_lateral_image = self.count_ill_pixels(self.lateral_image)
            return

        # Estrae il canali BGR
        blue_channel = img_bgr[:, :, 0]
        green_channel = img_bgr[:, :, 1]
        red_channel = img_bgr[:, :, 2]

        # Applica la condizione per trovare verde chiaro (piante potenzialmente malate)
        condition = (green_channel > 80) & (green_channel < 190) & \
                    (red_channel < 110) & (red_channel > 50) & \
                    (blue_channel < 70)

        light_green_count = np.count_nonzero(condition)

        # Calcola la percentuale
        total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
        percentage = (light_green_count / total_pixels) * 100

        if percentage > 3.0:
            self.get_logger().warn(f"WARNING: Light green detection high! Ratio: {percentage:.2f}%")

            if self.state == "SCAN":

                if self.turning_right:
                    self.ill_plant_detected_right = True
                
                if self.turning_left:
                    self.ill_plant_detected_left = True

        h, w = img_bgr.shape[:2]

        if self.image_w is None:
            self.image_w = w

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
    def count_ill_pixels(self, img_bgr):
        """
        Conta i pixel che rientrano nel range cromatico del "verde malato" 
        tramite filtraggio dei canali BGR.

        Returns:
            int: Numero di pixel rilevati.
        """
        
        # Estrae i canali RBG
        red_channel = img_bgr[:, :, 2]
        blue_channel = img_bgr[:, :, 0]
        green_channel = img_bgr[:, :, 1]

        # Condizione per identificare il verde malato
        condition = (green_channel > 80) & (green_channel < 190) & \
                    (red_channel < 110) & (red_channel > 50) & \
                    (blue_channel < 70)

        # Conta quanti pixel nell'immagine sono verde malato
        light_green_count = np.count_nonzero(condition)

        return light_green_count

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
        elif self.current_checkpoint == 4:
            goal_x = self.checkpoint5_x
            goal_y = self.checkpoint5_y
        elif self.current_checkpoint == 5:
            goal_x = self.checkpoint6_x
            goal_y = self.checkpoint6_y

        # Si allinea al prossimo checkpoint
        target_yaw = self.calculate_target_yaw(goal_x, goal_y)
        angle_err = self.angle_error(target_yaw)
            
        if abs(angle_err) > self.align_tol:
            # Ruota con controllo proporzionale
            angular_vel = max(min(0.8 * angle_err, self.turn_speed_max), -self.turn_speed_max)
            self.publish_twist(0.0, angular_vel)
        else:
            self.aligning = False

        return    

    def check_for_checkpoint_reached(self):

        # CHECKPOINT_1
        if not self.checkpoint1_reached and not self.checkpoint1_finished:
            d = self.distance_from_point(self.checkpoint1_x, self.checkpoint1_y)
            #print(d)
            if d <= 0.1:
                self.state = "CHECKPOINT_1"
                self.checkpoint1_reached = True

                if self.checkpoint_initial_yaw is None:
                    self.checkpoint_initial_yaw = self.yaw
                    
                self.publish_stop()
                return 
            
        # CHECKPOINT_2
        if not self.checkpoint2_reached and not self.checkpoint2_finished:
            d = self.distance_from_point(self.checkpoint2_x, self.checkpoint2_y)
            #print(d)
            if d <= 0.1:
                self.state = "CHECKPOINT_2"
                self.checkpoint2_reached = True

                if self.checkpoint_initial_yaw is None:
                    self.checkpoint_initial_yaw = self.yaw

                self.publish_stop()
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
        if not self.checkpoint4_reached and not self.checkpoint4_finished:
            d = self.distance_from_point(self.checkpoint4_x, self.checkpoint4_y)
            #print(d)
            if d <= 0.1:
                self.checkpoint4_reached = True
                self.checkpoint4_finished = True
                self.current_checkpoint += 1
                return       
            
        # CHECKPOINT_5
        if not self.checkpoint5_reached and not self.checkpoint5_finished:
            d = self.distance_from_point(self.checkpoint5_x, self.checkpoint5_y)
            #print(d)
            if d <= 0.1:
                self.state = "CHECKPOINT_5"
                print(self.state)
                self.checkpoint5_reached = True

                if self.checkpoint_initial_yaw is None:
                    self.checkpoint_initial_yaw = self.yaw

                self.publish_stop()
                return 
            
        # CHECKPOINT_6
        if not self.checkpoint6_reached:
            d = self.distance_from_point(self.checkpoint6_x, self.checkpoint6_y)
            #print(d)
            if d <= 0.1:
                self.stopped = True     # Ferma definitivamente il robot

                self.publish_stop()
                return 


    # -----------------------------
    # METODI DI STATO
    # -----------------------------
    def move_forward(self):

        """
        Gestisce la fase di avanzamento rettilineo durante la modalità di ricerca.
        
        Il metodo esegue le seguenti operazioni:
        1. Verifica che lo stato corrente sia effettivamente "FORWARD".
        2. Memorizza la posizione iniziale (odometria) al primo avvio della manovra.
        3. Calcola la distanza percorsa rispetto al punto di partenza.
        4. Se la distanza percorsa è uguale o superiore a 'search_straight_distance':
           - Arresta il robot.
           - Resetta tutti i parametri di navigazione e orientamento.
           - Passa allo stato "SCAN" per iniziare la rotazione di ricerca.
        5. Se la distanza non è stata ancora raggiunta, continua a pubblicare
           un comando di velocità lineare costante.
        """
        
        # Controllo di sicurezza: se lo stato interno non è FORWARD -> ritorna subito
        if self.state != "FORWARD": 
            return
        
        # Ruota verso il prossimo checkpoint (bivio)
        if self.aligning:
            self.align_to_next_checkpoint()
            return

        # Salva odometria di partenza  
        if self.search_start_x is None:
            self.search_start_y = self.y
            self.search_start_yaw = self.yaw
            self.search_start_x = self.x

        # Avanza per la distanza indicata, poi passa allo stato "SCAN"
        d = self.distance_from_point(self.search_start_x, self.search_start_y)

        if d >= self.search_straight_distance:
            self.publish_stop()

            # Resetta i parametri di scansione 
            self.turning_right = True
            self.turning_left = False
            self.realigning = False

            self.right_target_yaw = None
            self.left_target_yaw = None
            self.realigning_target_yaw = None

            # Resetta i parametri di stato
            self.search_start_x = None
            self.search_start_y = None
            self.search_start_yaw = None

            self.state = "SCAN"

            return

        # Altrimenti, prosegue dritto
        self.publish_twist(self.forward_speed, 0.0)

        return

    def scan(self):
        """
        Esegue una manovra di scansione angolare sul posto per cercare il target.
        
        La logica segue una sequenza di tre fasi:
        1. ROTAZIONE A DESTRA: Ruota il robot di 45 gradi verso destra rispetto 
           all'orientamento iniziale.
        2. ROTAZIONE A SINISTRA: Una volta completata la destra, ruota fino a 
           45 gradi a sinistra rispetto all'orientamento di partenza (arco totale di 90°).
        3. RIALLINEAMENTO: Torna all'orientamento originale memorizzato all'inizio 
           della scansione.
        
        Al termine del riallineamento, lo stato viene impostato su "FORWARD" per 
           proseguire l'esplorazione in linea retta se nessuna pianta potenzialmente malata
           è stata identificata; altrimenti, lo stato viene impostato su "ANALYZE" per
           analizzare un campione raccolto della pianta in questione.
        """
        # Controllo di sicurezza: se lo stato interno non è SCAN -> ritorna immediatamente
        if self.state != "SCAN":
            return

        # Gira a destra
        if self.turning_right:

            if self.right_target_yaw is None:
                self.start_spinning_yaw = self.yaw  # Salva l'orientamento corrente per il successivo riallineamento
                self.right_target_yaw = self.yaw - math.pi / 4

            # Se ha raggiunto l'ampiezza desiderata, smette di ruotare
            if abs(self.angle_error(self.right_target_yaw)) < self.align_tol:
                self.publish_stop()

                # Aggiorna le variabili interne
                self.right_target_yaw = None
                self.turning_left = True
                self.turning_right = False

                return
            
            # Altrimenti, ruota verso DESTRA
            self.publish_twist(0.0, -0.5)

            return
        
        # Gira a sinistra
        if self.turning_left:

            if self.left_target_yaw is None:
                self.left_target_yaw = self.start_spinning_yaw + math.pi / 4

            # Se ha raggiunto l'ampiezza desiderata, smette di ruotare
            if abs(self.angle_error(self.left_target_yaw)) < 0.05:
                self.publish_stop()
                self.left_target_yaw = None
                self.realigning = True
                self.turning_left = False

                return
            
            # Altrimenti, ruota verso SINISTRA
            self.publish_twist(0.0, 0.5)
            return
        
        # Riallineamento a orientamento di partenza
        if self.realigning:
            
            # Salva orientamento di partenza per la fase di riallineamento
            if self.realigning_target_yaw is None:
                self.realigning_target_yaw = self.start_spinning_yaw

            # Se ha completato il riallineamento: smette di ruotare e torna allo stato FORWARD
            if abs(self.angle_error(self.realigning_target_yaw)) < self.align_tol:
                self.publish_stop()
                self.realigning_target_yaw = None
                self.realigning = False

                # A questo punto il robot è orientato correttamente
                # Se sono state rilevate piante malate durante la scansione, analizza un campione delle foglie
                if self.ill_plant_detected_left or self.ill_plant_detected_right:
                    self.ill_plant_detected_left = False
                    self.ill_plant_detected_right = False

                    self.aligning = True

                    self.state = "ANALYZE"

                    return

                # Altrimenti procede dritto
                # Resetta le variabili relative allo stato FORWARD
                self.search_start_x = None
                self.search_start_y = None
                self.search_start_yaw = None

                self.aligning = True

                self.state = "FORWARD"

                return
            
            # Altrimenti, gira a destra
            self.publish_twist(0.0, -0.5)
            return

        return

    # Evitamento ostacoli semplificato adattato al contesto
    def avoid(self):

        if self.start_avoiding_yaw is None:
            self.start_avoiding_yaw = self.yaw
            return

        # Ruota di 90 gradi a sinistra per liberare il fronte
        angle_error = self.angle_error(self.start_avoiding_yaw + (self.avoiding_dir * math.pi/2))

        if abs(angle_error) >= 0.1:
            self.publish_twist(0.0, self.avoiding_dir * 0.3)

            return

        # Se la strada è libera, procede
        self.start_avoiding_yaw = None

        # Resetta le variabili di stato FORWARD
        self.search_start_x = None
        self.search_start_y = None
        self.search_start_yaw = None

        self.aligning = True

        self.state = "FORWARD"

        return

    def analyze(self):
        """
        Esegue l'analisi della pianta tramite classificazione SVM.

        Seleziona casualmente un'immagine dalla directory dati (70% Blight, 30% Healthy),
        estrae le feature HOG e utilizza il modello SVM per l'inferenza.
        Pubblica il risultato visuale e ripristina lo stato FORWARD.
        """
        self.publish_stop()

        # Configurazione percorsi e parametri
        IMG_SIZE = (128, 128)
        home = os.path.expanduser("~")
        base_path = os.path.join(home, "Agritech/workspace/Data/Original Data/")

        # 1. Selezione della cartella con probabilità pesata
        subfolder = random.choices(
            ["Leaf Blight", "Healthy"], 
            weights=[0.7, 0.3], 
            k=1
        )[0]
        
        folder_path = os.path.join(base_path, subfolder)

        # 2. Selezione di un'immagine randomica nella cartella scelta
        try:
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not files:
                raise FileNotFoundError(f"Nessun file immagine trovato in {folder_path}")
            
            img_path = os.path.join(folder_path, random.choice(files))
            img = cv2.imread(img_path)
            
            if img is None:
                raise ValueError(f"Impossibile leggere l'immagine: {img_path}")

        except Exception as e:
            self.get_logger().error(f"Errore caricamento immagine: {e}")

            # Reset variabili di stato e ripresa navigazione
            self.search_start_x = self.search_start_y = self.search_start_yaw = None
            self.aligning = True
            self.state = "FORWARD"
            return

        # Elaborazione immagine
        display_img = img.copy() 
        img_resized = cv2.resize(img, IMG_SIZE)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Inference HOG + SVM
        hog = cv2.HOGDescriptor((128,128), (32,32), (16,16), (16,16), 9)
        features = hog.compute(gray).flatten().reshape(1, -1)
        
        pred_idx = self.model.predict(features)[0]
        prob = self.model.predict_proba(features).max()
        label = self.class_names[pred_idx]

        # Overlay grafico dei risultati
        text = f"{label} ({prob*100:.1f}%)"
        cv2.putText(display_img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Pubblicazione immagine annotata
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(display_img, encoding="bgr8")
            self.pub_image.publish(annotated_msg)
            self.get_logger().info(f"Analisi completata [{subfolder}]: {text}")
        except Exception as e:
            self.get_logger().error(f"Errore pubblicazione immagine: {e}")

        # Reset variabili di stato e ripresa navigazione
        self.search_start_x = self.search_start_y = self.search_start_yaw = None
        self.aligning = True
        self.state = "FORWARD"
    
    def checkpoint(self, direction, check_num):
        """
        Gestisce la logica di ispezione ai checkpoint tramite rotazione e confronto visivo.

        Esegue una rotazione di 90° per acquisire un'immagine laterale, confronta il 
        numero di pixel "malati" tra la vista frontale e quella laterale per decidere 
        l'orientamento ottimale, quindi resetta lo stato per riprendere la navigazione.

        Args:
            direction (int): Direzione di rotazione (1 per sinistra, -1 per destra).
            check_num (int): Identificativo numerico del checkpoint corrente.
        """

        # Attende che l'immagine frontale sia memorizzata
        if self.frontal_image is None:
            return
        
        # Ruota di 90 gradi nella direzione specificata 
        angle_error = self.angle_error(self.checkpoint_initial_yaw + (direction*math.pi/2))

        if abs(angle_error) >= 0.1 and not self.rotated_at_checkpoint:
            self.publish_twist(0.0, 0.3*direction)
            print(f"Rotating: {self.rotated_at_checkpoint}")
            return
        
        if not self.rotated_at_checkpoint:
            self.rotated_at_checkpoint = True
            self.publish_stop()     # Smette di ruotare

        if self.lateral_image is None:
            return
        

        # Confronta le due immagini e decide la direzione da prendere
        if self.count_frontal_image is None:
            self.count_frontal_image = self.count_ill_pixels(self.frontal_image)
            return
        
        if self.count_lateral_image is None:
            self.count_lateral_image = self.count_ill_pixels(self.lateral_image)
            return

        print(f"FRONTAL IMAGE COUNT: {self.count_frontal_image}")
        print(f"LATERAL IMAGE COUNT: {self.count_lateral_image}")

        if self.count_frontal_image < self.count_lateral_image:
            self.publish_stop()     # Assicura che il robot non ruoti

        else:

            # Si riallinea alla direzione iniziale
            angle_error = self.angle_error(self.checkpoint_initial_yaw)

            if abs(angle_error) >= 0.1:
                self.publish_twist(0.0, -0.3*direction)
                return

        if check_num == 1:
            self.checkpoint1_reached = False      
            self.checkpoint1_finished = True
        elif check_num == 2:
            self.checkpoint2_reached = False      
            self.checkpoint2_finished = True     
        elif check_num == 5:
            self.checkpoint5_reached = False
            self.checkpoint5_finished = True

        # Resetta le variabili generiche di stato CHECKPOINT
        self.rotated_at_checkpoint = False
        self.frontal_image = None
        self.lateral_image = None
        self.count_frontal_image = None
        self.count_lateral_image = None   
        self.checkpoint_initial_yaw = None      

        # Incrementa il checkpoint corrente
        if self.current_checkpoint is None:
            self.current_checkpoint = 1
        else:
            self.current_checkpoint += 1 

        # Resetta le variabili relative allo stato FORWARD
        self.search_start_x = None
        self.search_start_y = None
        self.search_start_yaw = None

        self.aligning = True

        self.state = "FORWARD"
        return


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
        
        # Controlla se è a un bivio
        self.check_for_checkpoint_reached()

        # Evitamento ostacoli
        self.detect_obstacle()
        if self.obstacle_detected and self.state == "FORWARD":
            self.state = "AVOID"

        # Esegue la logica appropriata per lo stato corrente
        if self.state == "AVOID":
            self.avoid()
        elif self.state == "FORWARD": 
            self.move_forward()
        elif self.state == "SCAN":
            self.scan()
        elif self.state == "CHECKPOINT_1":
            self.checkpoint(direction=-1, check_num=1)
        elif self.state == "CHECKPOINT_2":
            self.checkpoint(direction=1, check_num=2)
        elif self.state == "CHECKPOINT_5":
            self.checkpoint(direction=1, check_num=5)    
        elif self.state == "ANALYZE":
            self.analyze()        

# -----------------------------
# MAIN
# -----------------------------
def main():
    rclpy.init()
    node = FiniteStateMachine()
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