import os
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from scipy.spatial import distance
from concurrent.futures import ProcessPoolExecutor

# Función para calcular el Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Parámetros para la detección de parpadeo
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# Inicializar contadores de parpadeo
parpadeo_total = 0
parpadeo_consec = 0

# Ruta a la imagen de referencia para la verificación de identidad
reference_image_path = "img/rostro_usuario.jpg"

# Verificación de la existencia de la imagen de referencia
def verificar_imagen_existe(image_path):
    if not os.path.exists(image_path):
        print(f"Error: La imagen {image_path} no existe.")
        return False
    return True

if not verificar_imagen_existe(reference_image_path):
    raise Exception(f"La imagen de referencia {reference_image_path} no existe.")

# Inicializar mediapipe para la detección de rostros y landmarks faciales
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Realizamos VideoCaptura
cap = cv2.VideoCapture(0)

# Inicializar ProcessPoolExecutor fuera del bucle
executor = ProcessPoolExecutor()

# Funciones para ejecutar en paralelo
def verificar_identidad(rostro_rgb, reference_image_path):
    try:
        result = DeepFace.verify(img1_path=rostro_rgb, img2_path=reference_image_path, model_name="Dlib", enforce_detection=False)
        verified = result["verified"]
        return f"Identidad Verificada: {verified}"
    except Exception as e:
        return f"Error: {str(e)}"

def analizar_emociones(rostro_rgb):
    try:
        emotion_analysis = DeepFace.analyze(img_path=rostro_rgb, actions=['emotion'], enforce_detection=False)
        dominant_emotion = emotion_analysis[0]['dominant_emotion'] if isinstance(emotion_analysis, list) else emotion_analysis['dominant_emotion']
        return f"Emoción: {dominant_emotion}"
    except Exception as e:
        return f"Error Emocion: {str(e)}"

frame_count = 0
identity_label = ""
emotion_label = ""

while True:
    # Leemos los frames
    ret, frame = cap.read()

    # Si hay error
    if not ret:
        break

    # Redimensionar frame para reducir la carga computacional
    frame = cv2.resize(frame, (640, 480))

    # Convertir a RGB una vez por frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar rostros
    face_detection_results = face_detection.process(frame_rgb)

    if face_detection_results.detections:
        for detection in face_detection_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame_rgb.shape
            x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            x2, y2 = int(bboxC.width * iw), int(bboxC.height * ih)

            # Validar que las dimensiones del ROI sean mayores que 0
            if x2 > 0 and y2 > 0:
                # Recorte del rostro
                rostro_rgb = frame_rgb[y1:y1+y2, x1:x1+x2]

                # Ejecutar verificación de identidad y análisis de emociones en paralelo cada 10 frames
                if frame_count % 10 == 0:
                    future_identity = executor.submit(verificar_identidad, rostro_rgb, reference_image_path)
                    future_emotion = executor.submit(analizar_emociones, rostro_rgb)

                    identity_label = future_identity.result()
                    emotion_label = future_emotion.result()

                # Dibujamos el rectángulo y las etiquetas en el frame
                cv2.rectangle(frame, (x1, y1), (x1 + x2, y1 + y2), (0, 0, 255), 2)
                cv2.putText(frame, identity_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, emotion_label, (x1, y1 + y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Detección de landmarks faciales y parpadeo usando mediapipe
                face_mesh_results = face_mesh.process(rostro_rgb)

                if face_mesh_results.multi_face_landmarks:
                    for face_landmarks in face_mesh_results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark
                        left_eye = [(landmarks[i].x * x2 + x1, landmarks[i].y * y2 + y1) for i in [33, 160, 158, 133, 153, 144]]
                        right_eye = [(landmarks[i].x * x2 + x1, landmarks[i].y * y2 + y1) for i in [362, 385, 387, 263, 373, 380]]
                        leftEAR = eye_aspect_ratio(left_eye)
                        rightEAR = eye_aspect_ratio(right_eye)
                        ear = (leftEAR + rightEAR) / 2.0

                        if ear < EYE_AR_THRESH:
                            parpadeo_consec += 1
                        else:
                            if parpadeo_consec >= EYE_AR_CONSEC_FRAMES:
                                parpadeo_total += 1
                            parpadeo_consec = 0

                        cv2.putText(frame, "Parpadeos: {}".format(parpadeo_total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Manejar el caso donde no se detectan rostros
        identity_label = "No se detecta rostro"
        emotion_label = ""
        cv2.putText(frame, identity_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # Mostrar el frame procesado
    cv2.imshow("DETECCION DE ROSTROS", frame)

    # Salir con 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
executor.shutdown()
