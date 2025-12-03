# FOR test

import time
import random

def init_system():
    print("[analog] Loading model")
    print("[analog] Opening the camera")
    return True

def get_current_emotion():
    # fake data for test
    return {
        "hasFace": True,
        "valence": round(random.uniform(-1, 1), 2),
        "arousal": round(random.uniform(-1, 1), 2),
        "intensity": round(random.uniform(0, 1), 2),
        "label": random.choice(["Happy", "Sad", "Neutral", "Angry"])
    }

def release_system():
    print("[analog] realse resource")