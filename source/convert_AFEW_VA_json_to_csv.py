import os
import json
import pandas as pd
import numpy as np
import math


PATH_DATA = '../data'
FOLDER_NAME = 'AFEW-VA'
FILE_NAME = 'AFEW-VAset.csv'

folder_path = os.path.join(PATH_DATA, FOLDER_NAME)
file_path = os.path.join(PATH_DATA, FILE_NAME)

if not os.path.exists(folder_path):
    print(f'Please unzip {FOLDER_NAME}.zip')
    exit()

# 定义 Morphset.csv 的列名结构
meta_columns = [
    'Filename', 'Dataset', 'Subject', 'Emotion Label',
    'Agreement', 'Arousal', 'Valence', 'Intensity'
]
x_columns = [f'x{i}' for i in range(68)]
y_columns = [f'y{i}' for i in range(68)]

all_columns = meta_columns + x_columns + y_columns

data_rows = []

# 用于统计坐标范围
all_xs = []
all_ys = []

# 假设文件夹是从 001 到 600
total_folders = 600
print(f"开始处理 {total_folders} 个文件夹...")

for i in range(1, total_folders + 1):
    folder_name = f"{i:03d}"
    file_name = f"{folder_name}.json"
    json_path = os.path.join(folder_path, folder_name, file_name)

    if not os.path.exists(json_path):
        continue

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = json.load(f)

        video_id = content.get('video_id', folder_name)
        frames = content.get('frames', {})
        sorted_frame_ids = sorted(frames.keys())

        for frame_id in sorted_frame_ids:
            frame_data = frames[frame_id]

            filename = f"{video_id}_{frame_id}.jpg"
            dataset_name = "NewData"
            subject = float(video_id)

            raw_arousal = float(frame_data.get('arousal', 0.0))
            raw_valence = float(frame_data.get('valence', 0.0))

            norm_arousal = float(raw_arousal / 10)
            norm_valence = float(raw_valence / 10)

            intensity = math.sqrt(norm_arousal ** 2 + norm_valence ** 2)
            intensity = min(1.0, intensity)

            emotion_label = "Unknown"
            agreement = 1.0

            # --- 3. 处理 Landmarks ---
            landmarks = frame_data.get('landmarks', [])

            if len(landmarks) != 68:
                continue

            landmarks_np = np.array(landmarks)
            xs = landmarks_np[:, 0]
            ys = landmarks_np[:, 1]

            row = [
                filename, dataset_name, subject, emotion_label,
                agreement, norm_arousal, norm_valence, intensity
            ]
            row.extend(xs)
            row.extend(ys)

            data_rows.append(row)

    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {str(e)}")

    if i % 50 == 0:
        print(f"已处理 {i}/{total_folders} 个文件夹...")

print(f"处理完成，生成 CSV，共 {len(data_rows)} 行...")

df = pd.DataFrame(data_rows, columns=all_columns)
df.to_csv(file_path, index=False)
print(f"成功保存: {FILE_NAME}")