# FOR test

import time
import random

#Loading model and Opening camera
#Return: True / False
def init_system():
    print("[analog] Loading model")
    print("[analog] Opening the camera")
    return True

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
    # fake data for test
    return {
        "hasFace": True,
        "valence": round(random.uniform(-1, 1), 2),
        "arousal": round(random.uniform(-1, 1), 2),
        "intensity": round(random.uniform(0, 1), 2),
        "label": random.choice(["Happy", "Sad", "Neutral", "Angry"])
    }

# Release camera and other resource
def release_system():
    print("[analog] realse resource")