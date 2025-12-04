# FOR test

import time
import cv2
from RealtimeEmotionAnalysis import FrameAnalyzer

global_cap = None
global_analyzer = None

#Loading model and Opening camera
#Return: True / False
def init_system():
    global global_cap, global_analyzer

    try:
        print("[analog] Loading model")
        global_analyzer = FrameAnalyzer(deadzone_threshold=0.05, calibration_frames=60)

        print("[analog] Opening the camera")
        global_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not global_cap.isOpened():
            return False
        print("Starting calibration...")
        print("=== PLEASE KEEP A NEUTRAL FACE ===")
        global_analyzer.start_calibration()
        while global_analyzer.is_calibrating():
            ret, frame = global_cap.read()
            if not ret:
                return False
            global_analyzer.process_frame(frame)
            time.sleep(0.01)
        return True

    except Exception as e:
        return False

# reading fram and analize
# Return: Dictionary
#     {
#         "hasFace": True,     
#         "valence": 0.5,       # (-1 ~ 1)
#         "arousal": 0.1,       #  (-1 ~ 1)
#         "intensity": 0.8,     #  (0 ~ 1)
#         "label": "Happy"      
#     }
# if face are not detected 
# Return: {"hasFace": False}
def get_current_emotion():
    global global_cap, global_analyzer

    no_face_result = {"hasFaces": False}

    ret, frame = global_cap.read()
    if not ret:
        return no_face_result

    try:
        result_data = global_analyzer.process_frame(frame)
    except Exception as e:
        return no_face_result

    if result_data is None:
        return no_face_result

    return {
        "hasFace": True,
        "valence": round(result_data['valence'], 2),
        "arousal": round(result_data['arousal'], 2),
        "intensity": round(result_data['intensity'], 2),
        "label": result_data['name']
    }

# Release camera and other resource
def release_system():
    global global_cap
    print("[analog] realse resource")
    if global_cap is not None:
        global_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if init_system():
        try:
            for i in range(20): # 测试读取20次
                data = get_current_emotion()
                print(f"Test {i}: {data}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            release_system()