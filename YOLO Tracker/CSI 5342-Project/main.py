import os
import cv2
import math
import numpy as np
import supervision as sv
from collections import deque
from ultralytics import YOLO
from scipy.spatial import KDTree
from trackers import ByteTrackTracker

CONFIDENCE   = 0.25         
HISTORY_SIZE = 21      
model = YOLO("./yolo-model-weights/best.pt")      
    
def load_file(file_path):
    try:
        class_id = int(file_path.split()[-2])
        data = np.loadtxt(file_path, delimiter=',', dtype=int)
        label = np.full(len(data), class_id)
        return data,label
    except Exception as e:
        print(f"{e}")
        return None
        

def load_all(dir_path):
    all_points_list = []
    all_labels_list = []
    label_index = {}
    sum_index = 0
    for root, dirs, files in os.walk(dir_path):
        for file in files:         
            file_path = os.path.join(root, file)
            data, labels = load_file(file_path)
            all_points_list.append(data)
            all_labels_list.append(labels)
            label_index[int(labels[0])] = sum_index
            sum_index += len(data)
    all_points = np.vstack(all_points_list)    
    all_labels = np.concatenate(all_labels_list)
    tree = KDTree(all_points)
    return tree,all_labels,label_index 
            
def detect_path(cx, cy, tree, labels, label_index):
    dist, index = tree.query([cx, cy])
    if dist < 25:
        return labels[index], index - label_index[labels[index]]
    else:
        return -1,-1


def calculate_speed_and_predict(tid, cx, cy, history):
    if tid == -1:
        return -1,None
        
    if tid not in history:
        history[tid] = deque(maxlen=HISTORY_SIZE)
    history[tid].append((cx, cy))
    
    if len(history[tid]) < HISTORY_SIZE:
        return -1, None
    
    samples = [history[tid][i] for i in range(-1, -22, -4)]
    
    deltas = [((samples[i][0] - samples[i+1][0]) / 4.0, 
               (samples[i][1] - samples[i+1][1]) / 4.0) 
              for i in range(len(samples) - 1)]
    
    avg_vx = sum(d[0] for d in deltas) / len(deltas)
    avg_vy = sum(d[1] for d in deltas) / len(deltas)
    
    speed_px_frame = np.sqrt(avg_vx**2 + avg_vy**2)
    
    predictions = [(int(cx + avg_vx * p * 4), int(cy + avg_vy * p * 4)) 
                   for p in range(1, 6)]
        
    return speed_px_frame, predictions
    

def semi_trans_video(SOURCE_VIDEO, OUTPUT_VIDEO, tree, labels, label_index, show =True):
    tracker = ByteTrackTracker()
    tracker.reset()
    
    rec = {}
    history = {}
    
    cap = cv2.VideoCapture(SOURCE_VIDEO)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    print(f"Processing: {SOURCE_VIDEO}")
    collision_flag = False
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        all_predictions = {}
        car_size = {}
        
        collision_risk = 0
        result     = model(frame, conf=CONFIDENCE, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update(detections)   
        
        annotated_frame = frame.copy()
        
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )

        for i in range(len(detections)):
            tid  = detections.tracker_id[i]                  
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if tid != -1:
                car_size[tid] = math.sqrt(((x2-x1)//2)**2 + ((y2-y1)//2)**2)
            velocity,predicts = calculate_speed_and_predict(tid, cx, cy, history)
            path_label,frame_index = detect_path(cx, cy, tree, labels, label_index)
            
            d = rec.get(tid, {})
            d[path_label] = d.get(path_label, 0) + 1
            rec[tid] = d
            max_path = max(d, key=d.get)
            
            if predicts is not None:
                for pt in predicts:
                    cv2.circle(annotated_frame, pt, 3, (0, 0, 255), -1) 
                if collision_risk < 5:
                    for a_tid in all_predictions.keys():
                        if a_tid != tid:
                            a_predicts = all_predictions[a_tid]
                            for i in range(min(len(a_predicts), len(predicts))):
                                distance = np.linalg.norm(np.array(a_predicts[i]) - np.array(predicts[i]))
                                if distance < (car_size[a_tid]+car_size[tid])*0.5:
                                    collision_risk = max(collision_risk, 5-i)
                                    if collision_risk == 5:
                                        collision_flag = True
                                        
                                    
                all_predictions[tid] = predicts
                
            v_text = f"{velocity:.1f}px/f" if velocity > 0 else "Calibrating..."    
            text = f"tid:{tid} | path:{max_path} | index:{frame_index} | {v_text}"
            cv2.putText(annotated_frame, text, (cx, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 
        risk = 20*collision_risk   
        cv2.putText(annotated_frame, f"collision:{risk}%", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)            
                           
            
        if show:
            cv2.imshow("Reconstructed", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.write(annotated_frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return collision_flag

if __name__=="__main__":
    Output_ROOT = './Output_Video'
    if not os.path.exists(Output_ROOT):
        os.makedirs(Output_ROOT)
        
    tree,labels,label_index = load_all('./pixel_path_coordinates') 
    

    # print(semi_trans_video('./Project_Input_Videos/video_data_collision_13.mp4', './Output_Video/video_data_collision_13.mp4', tree, labels, label_index, show=True))
    
    for root, dirs, files in os.walk('./Project_Input_Videos'):
        for file in files:
            SOURCE_VIDEO = os.path.join(root, file)
            OUTPUT_VIDEO = os.path.join(Output_ROOT, file)
            print(semi_trans_video(SOURCE_VIDEO, OUTPUT_VIDEO, tree, labels, label_index, show=False))