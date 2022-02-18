import cv2
import mediapipe as mp
import numpy as np
from numpy import linalg as LA
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import time
import math
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

framenumber = []

hip_position = []
shoulder_position = []
wrist_position = []

wrist_angle_data = []
body_turn_data = []
elbow_angle_data = []
hip_turn_data = []


#start = False
start = True
def calculate_body_turn(shoulder_vec, hip_vec):
    v1=shoulder_vec/LA.norm(shoulder_vec)
    v2=hip_vec/LA.norm(hip_vec)
    res = np.dot(v1, v2)
    angle_rad = np.arccos(res)
    return math.degrees(angle_rad)


def calculate_angle_2d(start, middle, end):
    v1 = np.array([start[0] - middle[0], start[1] - middle[1]])
    v2 = np.array([end[0] - middle[0], end[1] - middle[1]])
    v1mag = np.sqrt(v1[0]**2 + v1[1]**2)
    v2mag = np.sqrt(v2[0]**2 + v2[1]**2)
    v1norm = np.array([v1[0]/v1mag, v1[1]/v1mag])
    v2norm = np.array([v2[0]/v2mag, v2[1]/v2mag])
    res = np.dot(v1norm, v2norm)
    angle_rad = np.arccos(res)
    return math.degrees(angle_rad)

def calculate_angle_3d(start, middle, end):
    v1 = np.array([start[0] - middle[0], start[1] - middle[1], start[2]-middle[2]])
    v2 = np.array([end[0] - middle[0], end[1] - middle[1], end[2] - middle[2]])
    v1mag = LA.norm(v1)
    v2mag = LA.norm(v2)
    v1norm = np.array([v1[0]/v1mag, v1[1]/v1mag, v1[2]/v1mag])
    v2norm = np.array([v2[0]/v2mag, v2[1]/v2mag, v2[1]/v2mag])
    res = np.dot(v1norm, v2norm)
    angle_rad = np.arccos(res)
    return math.degrees(angle_rad)
    

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence = 0.9) as pose:
    currentframe = 0
    cap = cv2.VideoCapture('slowmocropped2.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("can't receive frame. exiting")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            if start:
                #right left hip landmark
                rhip_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                lhip_landmark = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

                #right left shoulder landmark
                rshoulder_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                lshoulder_landmark = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                #print(rshoulder_landmark)

                #right wrist landmark
                rwrist_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                #calculate elbow angle
                elbow_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                elbow_pos_np = np.array([elbow_landmark.x, elbow_landmark.y, elbow_landmark.y])
                shoulder_pos_np = np.array([rshoulder_landmark.x, rshoulder_landmark.y, rshoulder_landmark.z])
                wrist_pos_np = np.array([rwrist_landmark.x, rwrist_landmark.y, rwrist_landmark.z])
                elbow_angle = calculate_angle_2d(shoulder_pos_np, elbow_pos_np, wrist_pos_np)
                
                #calculate wrist angle
                hand_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]#right index finger landmark values
                hand_pos_np = np.array([hand_landmark.x, hand_landmark.y, hand_landmark.z])
                wrist_angle = calculate_angle_2d(elbow_pos_np, wrist_pos_np, hand_pos_np)

                #calculate body turn
                hip_vec = np.array([rhip_landmark.x - lhip_landmark.x, rhip_landmark.z - lhip_landmark.z])
                shoulder_vec = np.array([rshoulder_landmark.x - lshoulder_landmark.x, rshoulder_landmark.z - lshoulder_landmark.z])
                body_turn_angle = calculate_body_turn(shoulder_vec, hip_vec)

                #calculate hip turn
                rfoot_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
                lfoot_landmark = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
                
                foot_vec = np.array([rfoot_landmark.x - lfoot_landmark.x, rfoot_landmark.z - lfoot_landmark.z])
                hip_turn_angle = calculate_body_turn(foot_vec, hip_vec)

                #capture data
                hip_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
                shoulder_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                wrist_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

                wrist_angle_data.append(wrist_angle)
                hip_turn_data.append(hip_turn_angle)

                body_turn_data.append(body_turn_angle)
                elbow_angle_data.append(elbow_angle)

                currentframe += 1
                framenumber.append(currentframe)

        except:
            pass
        
        #make detection
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness = 2, circle_radius = 2),
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness = 2, circle_radius = 2) )

        cv2.imshow('Mediapipe feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("shoulder pos : ", shoulder_pos_np)
            print("elbow pos np : ", elbow_pos_np)
            print("wrist pos np: ", wrist_pos_np)
            mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
            break

        #if esc is pressed, start calculating 
        if cv2.waitKey(5) & 0xFF == 27:
            start = True
            print('starting recording')
            time.sleep(2)
    cap.release()
    cv2.destroyAllWindows()


    hip_pos = np.zeros((3, len(hip_position)))
    shoulder_pos = np.zeros((3, len(shoulder_position)))
    wrist_pos = np.zeros((3, len(wrist_position)))
    index=0
    print(np.shape(hip_turn_data))

    for hip_data in hip_position:
        new_col = [hip_data.x, hip_data.y, hip_data.z]
        hip_pos[:, index] = new_col
        index+=1
    index = 0
    for shoulder_data in shoulder_position:
        new_col = [shoulder_data.x, shoulder_data.y, shoulder_data.z]
        shoulder_pos[:, index] = new_col
        index+=1
    index=0
    for wrist_data in wrist_position:
        new_col = [wrist_data.x, wrist_data.y, wrist_data.z]
        wrist_pos[:, index] = new_col
        index+=1

    #if smooth, aka framerate always increments by 1, then ok to do gradient
    smooth = True
    for x in range(len(framenumber)-1):
        if framenumber[x+1]-framenumber[x]==1:
            continue
        else:
            smooth = False
            break
    
    #if data is smooth, calculate acceleration
    if smooth:
        hip_accel = np.gradient(np.gradient(hip_pos, axis=1), axis=1)
        shoulder_accel = np.gradient(np.gradient(shoulder_pos, axis=1), axis=1)
        wrist_accel = np.gradient(np.gradient(wrist_pos, axis=1), axis=1)
        
        hip_accel_norm = LA.norm(hip_accel, axis=0)
        shoulder_accel_norm = LA.norm(shoulder_accel, axis=0)
        wrist_accel_norm = LA.norm(wrist_accel, axis=0)

        torso_angular_accel = np.gradient(np.gradient(body_turn_data))
        hip_angular_accel = np.gradient(np.gradient(hip_turn_data))
        wrist_angular_accel = np.gradient(np.gradient(wrist_angle_data))

        torso_angular_accel_max = np.argmax(torso_angular_accel)
        hip_angular_accel_max = np.argmax(hip_angular_accel)
        wrist_angular_accel_max = np.argmax(wrist_angular_accel)


        print("highest torso angular accel at frame ", torso_angular_accel_max)
        print("highest hip angular accel at frame ", hip_angular_accel_max)
        print("highest wrist angular accel at  ", wrist_angular_accel_max)

        hip_turn_data_smooth = savgol_filter(hip_turn_data, 51, 3)
        hip_angular_accel_smooth = np.gradient(np.gradient(hip_turn_data_smooth))

        hip_angular_accel_graph = plt.figure(1)
        plt.scatter(framenumber, hip_angular_accel)
        plt.title("hip angular acceleration - no smoothing")

        hip_angular_accel_graph_smooth = plt.figure(7)
        plt.scatter(framenumber, hip_angular_accel_smooth)
        plt.title("hip angular acceleration with smoothing")
        
        torso_angular_accel_graph = plt.figure(2)
        plt.scatter(framenumber, torso_angular_accel)
        plt.title("torso angular acceleration - rotation")

        wrist_angular_accel_graph = plt.figure(3)
        plt.scatter(framenumber, wrist_angular_accel)
        plt.title("wrist angular acceleration")

        elbow_angle_graph = plt.figure(4)
        plt.scatter(framenumber, elbow_angle_data)
        plt.title("elbow angle")

        body_turn_graph = plt.figure(5)
        plt.scatter(framenumber, hip_turn_data_smooth)
        plt.title("hip turn with smoothing")

        hip_turn_graph = plt.figure(6)
        plt.scatter(framenumber, hip_turn_data)
        plt.title("hip turn no smoothing")

        plt.show()
    else:
        print('camera missed some frames and was not able to do analysis')





    