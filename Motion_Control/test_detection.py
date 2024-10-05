import cv2
from ultralytics import YOLO
import time
import math

# Konstanta
MODEL_PATH = 'models/best.pt'
# VIDEO_PATH = 0
VIDEO_PATH = "Dataset/Video/B.mp4"
FRAME_WIDTH, FRAME_HEIGHT = 640, 640

# Dictionary untuk mapping class ID ke nama
class_names = {0: 'green_ball', 1: 'red_ball'}

# List untuk menyimpan warna berdasarkan class ID
class_colors = [
    (0, 255, 0),  # Hijau untuk green_ball (ID 0)
    (0, 0, 255)   # Merah untuk red_ball (ID 1)
]

# Hitung titik tengah frame
center_x_frame = FRAME_WIDTH // 2
left_x_frame = 100
right_x_frame = 540

# Titik acuan pivot
pivot_point = (FRAME_WIDTH // 2, FRAME_HEIGHT)
bottom_pivot = 340
fps = 0.00


def draw_center_line(frame, fps, green, red, mid):
    """
    Fungsi untuk menggambar garis tengah berwarna putih pada frame.
    """
    # Tampilkan FPS di frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if green is not None:
        green_center_x, green_center_y = green[4]
        cv2.putText(frame, f'Green: ({green_center_x:.0f}, {green_center_y:.0f})', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if red is not None:
        red_center_x, red_center_y = red[4]
        cv2.putText(frame, f'Red: ({red_center_x:.0f}, {red_center_y:.0f})', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.putText(frame, f'Mid: ({mid[0]:.0f}, {mid[1]:.0f})', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Garis vertikal di tengah frame
    cv2.line(frame, (center_x_frame, 0),
             (center_x_frame, FRAME_HEIGHT), (255, 255, 255), 2)

    # Garis vertikal di kiri frame
    cv2.line(frame, (left_x_frame, 0),
             (left_x_frame, FRAME_HEIGHT), (255, 255, 255), 2)

    # Garis vertikal di kanan frame
    cv2.line(frame, (right_x_frame, 0),
             (right_x_frame, FRAME_HEIGHT), (255, 255, 255), 2)

    # Garis horizontal di kanan frame
    cv2.line(frame, (0, bottom_pivot),
             (FRAME_WIDTH, bottom_pivot), (255, 255, 255), 2)


def calculate_distance(point1, point2):
    """
    Fungsi untuk menghitung jarak Euclidean antara dua titik.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_mid_point(point1, point2):
    """
    Fungsi untuk menghitung titik tengah antara dua titik.
    """
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)


def main():

    # urgent
    mid_point = 320, 320

    # Load YOLOv8 model
    model = YOLO(MODEL_PATH)

    # Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()  # Ambil frame dari video
        if not ret:
            break

        # Resize frame to 640x640
        resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Catat waktu sebelum deteksi
        start_time = time.time()

        # Menggunakan model.predict() dengan source dari resized_frame dan imgsz 640
        results = model.predict(source=resized_frame,
                                imgsz=640, conf=0.6, iou=0.7, max_det=8, device='cpu')

        # Catat waktu setelah deteksi
        end_time = time.time()

        # Hitung waktu yang berlalu per frame dan FPS
        elapsed_time = end_time - start_time
        fps = 1 / elapsed_time

        # Variabel untuk menyimpan objek terdekat per kelas
        closest_green_ball = None
        closest_red_ball = None
        closest_green_distance = float('inf')
        closest_red_distance = float('inf')

        # Ekstraksi manual dari bounding box, confidence scores, dan class labels
        for result in results:
            for box in result.boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                # Hitung titik tengah dari bounding box
                center_x_box = (x1 + x2) / 2
                center_y_box = (y1 + y2) / 2
                center_box = (center_x_box, center_y_box)

                # Tentukan warna berdasarkan class ID menggunakan class_colors
                color = class_colors[cls] if cls < len(
                    class_colors) else (255, 255, 255)

                # Print detected object details
                detected_class_name = class_names.get(cls, 'Unknown')
                print(
                    f"Detected object: {detected_class_name}, Confidence: {conf:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")

                # Hitung jarak ke pivot point
                distance_to_pivot = calculate_distance(center_box, pivot_point)

                # Filter berdasarkan posisi Y (harus di atas bottom_pivot)
                if center_y_box < bottom_pivot:
                    # Jika class 'green_ball', periksa jarak dan simpan yang terdekat
                    if cls == 0 and distance_to_pivot < closest_green_distance:
                        closest_green_distance = distance_to_pivot
                        closest_green_ball = (
                            x1, y1, x2, y2, center_box, color)

                    # Jika class 'red_ball', periksa jarak dan simpan yang terdekat
                    if cls == 1 and distance_to_pivot < closest_red_distance:
                        closest_red_distance = distance_to_pivot
                        closest_red_ball = (x1, y1, x2, y2, center_box, color)

                # Gambarkan kotak deteksi dan teks dengan warna yang sesuai
                cv2.rectangle(resized_frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), color, 2)
                label = f'{detected_class_name} {conf:.2f}'
                cv2.putText(resized_frame, label, (int(x1), int(
                    y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Jika ada green_ball dan red_ball terdekat
        if closest_green_ball is not None and closest_red_ball is not None:
            # Dapatkan titik tengah dari kedua bola
            _, _, _, _, green_center, _ = closest_green_ball
            _, _, _, _, red_center, _ = closest_red_ball

            cv2.circle(resized_frame, (int(green_center[0]), int(
                green_center[1])), 5, (0, 255, 0), -1)  # Titik hijau

            cv2.circle(resized_frame, (int(red_center[0]), int(
                red_center[1])), 5, (0, 0, 255), -1)  # Titik merah

            # Hitung titik tengah di antara kedua bola
            mid_point = calculate_mid_point(green_center, red_center)

            # Gambar garis yang menghubungkan titik tengah green_ball dan red_ball
            cv2.line(resized_frame, (int(green_center[0]), int(green_center[1])),
                     (int(red_center[0]), int(red_center[1])), (0, 255, 255), 2)

            # Gambar garis dari titik tengah dua bola ke pivot_point
            cv2.line(resized_frame, (int(mid_point[0]), int(mid_point[1])),
                     pivot_point, (0, 255, 255), 2)

            # Gambarkan titik biru di tengah-tengah antara green_ball dan red_ball
            cv2.circle(resized_frame, (int(mid_point[0]), int(mid_point[1])),
                       5, (255, 0, 0), -1)  # Titik biru

        # Gambar garis tengah pada frame
        draw_center_line(resized_frame, fps, closest_green_ball,
                         closest_red_ball, mid_point)

        # Display the frame with the detections
        cv2.imshow('ASV-KKI 2024', resized_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
