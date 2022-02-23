import cv2
import mediapipe as mp
import numpy as np
from numpy import linalg as LA
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import time
import math
from pandas import DataFrame

#CHANGE VIDEO NAME 
#프로그렘 돌리기전에 파일 이름 바꿔야됨
#파일 타입 무조건 포함해야됨 예) ".mp4"
video_name = "test99trim.mp4"


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

prev_elbow_angle = 0
prev_wrist_angle = 0

swing_start_frame = 0
swing_end_frame = 0
prev_wrist_max_pos = 0
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
do_once = True
def calculate_body_turn(shoulder_vec, hip_vec):
    v1=shoulder_vec/LA.norm(shoulder_vec)
    v2=hip_vec/LA.norm(hip_vec)
    res = np.dot(v1, v2)
    angle_rad = np.arccos(res)
    if(np.cross(v1, v2)<0):
        return -1*math.degrees(angle_rad)
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
    if(res>1):
        res=1
    elif(res<-1):
        res=-1
    angle_rad = np.arccos(res)
    return math.degrees(angle_rad)
    

with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence = 0.5, enable_segmentation = True) as pose:
    currentframe = 0
    cap = cv2.VideoCapture(video_name)
    
    while cap.isOpened():
        width  = cap.get(3)
        height = cap.get(4)
        ret, frame = cap.read()
        
        if not ret:
            print("can't receive frame. exiting")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        #extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            if do_once:
                #calculate global x axis once
                global_x_right = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
                global_x_left = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
                global_x_vec = np.array([global_x_right.x - global_x_left.x, global_x_right.z - global_x_left.z])

                do_once=False
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
                lwrist_landmark = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

                if(lwrist_landmark.visibility > 0.8):#only calculate elbow and wrist angle is wrist is visible
                    #calculate elbow angle
                    relbow_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    lelbow_landmark = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    elbow_pos_np = np.array([lelbow_landmark.x, lelbow_landmark.y, lelbow_landmark.z]) #left elbow
                    shoulder_pos_np = np.array([lshoulder_landmark.x, lshoulder_landmark.y, lshoulder_landmark.z]) #left shoulder
                    wrist_pos_np = np.array([lwrist_landmark.x, lwrist_landmark.y, lwrist_landmark.z]) #left wrist
                    elbow_angle = calculate_angle_3d(shoulder_pos_np, elbow_pos_np, wrist_pos_np)
                    
                    #calculate wrist angle
                    hand_landmark = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]#left index finger landmark values
                    hand_pos_np = np.array([hand_landmark.x, hand_landmark.y, hand_landmark.z])#left hand
                    wrist_angle = calculate_angle_3d(elbow_pos_np, wrist_pos_np, hand_pos_np)#angle of left wrist

                    prev_elbow_angle = elbow_angle
                    prev_wrist_angle = wrist_angle
                else:
                    print('elbow angle ', elbow_angle)
                    elbow_angle = prev_elbow_angle
                    wrist_angle = prev_wrist_angle



                lfoot_landmark = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
                rfoot_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
                #foot_vec = np.array([global_x_right.x - global_x_left.x, global_x_right.z - global_x_left.z])
                foot_vec = np.array([rfoot_landmark.x - lfoot_landmark.x, rfoot_landmark.z - lfoot_landmark.z])
                #calculate hip turn
                hip_vec = np.array([rhip_landmark.x - lhip_landmark.x, rhip_landmark.z - lhip_landmark.z])
                #foot_vec = global_x_vec
                hip_turn_angle = calculate_body_turn(foot_vec, hip_vec)

                #calculate body turn
                shoulder_vec = np.array([rshoulder_landmark.x - lshoulder_landmark.x, rshoulder_landmark.z - lshoulder_landmark.z])
                body_turn_angle = calculate_body_turn(hip_vec, shoulder_vec)
                

                #capture datas
                hip_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
                shoulder_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                wrist_position.append(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

                wrist_angle_data.append(wrist_angle)
                hip_turn_data.append(hip_turn_angle)

                body_turn_data.append(body_turn_angle)
                elbow_angle_data.append(elbow_angle)
                #print("hip angle ", hip_turn_angle)
                #left_ankle_coord = (int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x * width), int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y * height))
                #right_foot_coord = (int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * width), int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * height))
                #print("right foot y pos ", rfoot_landmark.y)
                
                if((landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y < landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y) and swing_start_frame==0):
                    swing_start_frame = currentframe
                #time.sleep(0.1)
                #print(lwrist_landmark.x)
                if(prev_wrist_max_pos < lwrist_landmark.x):
                    #print('end frame update')
                    prev_wrist_max_pos = lwrist_landmark.x
                    swing_end_frame = currentframe
                #image = cv2.circle(image, left_ankle_coord, 20, (255, 0, 0), 3)
                #image = cv2.circle(image, right_foot_coord, 20, (0, 255, 0), 3)
                #image = cv2.circle(image, (width*0.1, height*0.9), 20, (255,0,0), 3)
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
        """
        hip_accel = np.gradient(np.gradient(hip_pos, axis=1), axis=1)
        shoulder_accel = np.gradient(np.gradient(shoulder_pos, axis=1), axis=1)
        wrist_accel = np.gradient(np.gradient(wrist_pos, axis=1), axis=1)
        
        hip_accel_norm = LA.norm(hip_accel, axis=0)
        shoulder_accel_norm = LA.norm(shoulder_accel, axis=0)
        wrist_accel_norm = LA.norm(wrist_accel, axis=0)
        """

        torso_angular_vel = np.gradient(body_turn_data)
        hip_angular_vel = np.gradient(hip_turn_data)
        wrist_angular_vel = np.gradient(wrist_angle_data)
        elbow_angular_vel = np.gradient(elbow_angle_data)

        
        torso_angular_vel_max = np.argmax(torso_angular_vel)
        hip_angular_vel_max = np.argmax(hip_angular_vel)
        wrist_angular_vel_max = np.argmax(wrist_angular_vel)

        hip_turn_data_smooth = savgol_filter(hip_turn_data, 20, 3)
        hip_angular_vel_smooth = np.gradient(hip_turn_data_smooth, 3)

        torso_turn_data_smooth = savgol_filter(body_turn_data, 20, 3)
        torso_angular_vel_smooth = np.gradient(torso_turn_data_smooth, 3)

        wrist_angle_data_smooth = savgol_filter(wrist_angle_data, 20, 3)
        wrist_angular_vel_smooth = np.gradient(wrist_angle_data_smooth, 10)

        elbow_angle_data_smooth = savgol_filter(elbow_angle_data, 20, 3)
        elbow_angular_vel_smooth = np.gradient(elbow_angle_data_smooth, 3)

        swing_length = swing_end_frame - swing_start_frame

        print("엉덩관절 최대 회전 각속도 @ frame number ", np.argmax(hip_angular_vel_smooth[swing_start_frame : swing_end_frame]))
        print((np.argmax(hip_angular_vel_smooth[swing_start_frame : swing_end_frame]) - swing_start_frame)/swing_length * 100.0, "%")

        print("몸통 최대 각속도 @ frame number ", np.argmax(torso_angular_vel_smooth[swing_start_frame : swing_end_frame]))
        print((np.argmax(torso_angular_vel_smooth[swing_start_frame : swing_end_frame]) - swing_start_frame)/swing_length * 100.0, "%")

        print("팔꿈치 최대 각속도 @ frame number", np.argmax(elbow_angular_vel_smooth[swing_start_frame : swing_end_frame]))
        print((np.argmax(elbow_angular_vel_smooth[swing_start_frame : swing_end_frame]) - swing_start_frame)/swing_length * 100.0, "%")
        
        print("손목 최대 각속도 @ frame number ", np.argmax(wrist_angular_vel_smooth[swing_start_frame : swing_end_frame]))
        print((np.argmax(wrist_angular_vel_smooth[swing_start_frame : swing_end_frame]) - swing_start_frame)/swing_length * 100.0, "%")

        print("swing start at frame ", swing_start_frame)
        print('swing end at frame ', swing_end_frame)


        df = DataFrame({'Frame': framenumber, 
                        'hip angle no smoothing': hip_turn_data, 'hip angle with smoothing': hip_turn_data_smooth, 'hip angular velocity (from smooth)': hip_angular_vel_smooth,
                        'torso angle no smoothing': body_turn_data, 'torso angle with smoothing': torso_turn_data_smooth, 'torso angular velocity (from smooth)': torso_angular_vel_smooth,
                        'wrist angle no smoothing': wrist_angle_data, 'wrist angle with smoothing': wrist_angle_data_smooth, 'wrist angular velocity (from smooth)' : wrist_angular_vel_smooth,
                        'elbow angle no smoothing': elbow_angle_data, 'elbow angle with smoothing': elbow_angle_data_smooth, 'elbow angular velocity(from smooth)': elbow_angular_vel_smooth})
        df.to_excel('results.xlsx', sheet_name='sheet1', index=False)

        fig, axs = plt.subplots(3 ,4)
        axs[0,0].plot(framenumber, hip_turn_data_smooth)
        axs[0,0].set_title('hip turn smooth')

        axs[1,0].plot(framenumber, hip_turn_data)
        axs[1,0].set_title('hip turn')

        axs[2,0].plot(framenumber, hip_angular_vel_smooth)
        axs[2,0].set_title('hip angular velocity')


        axs[0,1].plot(framenumber, torso_turn_data_smooth)
        axs[0,1].set_title('torso turn smooth')

        axs[1,1].plot(framenumber, body_turn_data)
        axs[1,1].set_title('torso turn')

        axs[2,1].plot(framenumber, torso_angular_vel_smooth)
        axs[2,1].set_title('torso angular velocity')


        axs[0,2].plot(framenumber, elbow_angle_data_smooth)
        axs[0,2].set_title('elbow angle smooth')

        axs[1,2].plot(framenumber, elbow_angle_data)
        axs[1,2].set_title('elbow angle')

        axs[2,2].plot(framenumber, elbow_angular_vel_smooth)
        axs[2,2].set_title('elbow angular velocity')


        axs[0,3].plot(framenumber, wrist_angle_data_smooth)
        axs[0,3].set_title('wrist angle smooth')

        axs[1,3].plot(framenumber, wrist_angle_data)
        axs[1,3].set_title('wrist angle')

        axs[2,3].plot(framenumber, wrist_angular_vel_smooth)
        axs[2,3].set_title('wrist angular velocity')

        """
        hip_angvel_graph = plt.figure(0)
        plt.scatter(framenumber, hip_angular_vel_smooth)
        plt.title("hip angular velocity with smoothing")
        
        torso_angvel_graph = plt.figure(1)
        plt.scatter(framenumber, torso_angular_vel_smooth)
        plt.title("torso angular velocity with smoothing")

        wrist_angvel_graph = plt.figure(2)
        plt.scatter(framenumber, wrist_angular_vel_smooth)
        plt.title("wrist angular velocity with smoothing")

        elbow_angvel_graph = plt.figure(3)
        plt.scatter(framenumber, elbow_angular_vel_smooth)
        plt.title("elbow angular velocity with smoothing")

        hip_turn_graph = plt.figure(4)
        plt.scatter(framenumber, hip_turn_data_smooth)
        plt.scatter(framenumber, hip_turn_data)
        plt.title("hip turn blue = smoothed")

        torso_turn_graph = plt.figure(6)
        plt.scatter(framenumber, torso_turn_data_smooth)
        plt.scatter(framenumber, body_turn_data)
        plt.title("torso turn blue = smoothed")

        elbow_angle_graph = plt.figure(7)
        plt.scatter(framenumber, elbow_angle_data_smooth)
        plt.scatter(framenumber, elbow_angle_data)
        plt.title("elbow angle data blue = smoothed")

        wrist_angle_graph = plt.figure(8)
        plt.scatter(framenumber, wrist_angle_data_smooth)
        plt.scatter(framenumber, wrist_angle_data)
        plt.title("wrist angle data blue = smoothed")
        """
        plt.show()
    else:
        print('camera missed some frames and was not able to do analysis')
