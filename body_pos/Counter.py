import  numpy as np

def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180

    cross = v2[0] * v1[1] - v2[1] * v1[0]
    if cross < 0:
        angle = - angle
    return angle

def train_counter(keypoints, type):
    global flag
    keypoints = np.array(keypoints)
    Counter = 0
    v1 = keypoints[5] - keypoints[6]
    v2 = keypoints[8] - keypoints[6]
    angle_right_arm = get_angle(v1, v2)

    # 计算左臂与水平方向的夹角
    v1 = keypoints[7] - keypoints[5]
    v2 = keypoints[6] - keypoints[5]
    angle_left_arm = get_angle(v1, v2)

    # 计算右肘的夹角
    v1 = keypoints[6] - keypoints[8]
    v2 = keypoints[10] - keypoints[8]
    angle_right_elbow = get_angle(v1, v2)

    # 计算左肘的夹角
    v1 = keypoints[5] - keypoints[7]
    v2 = keypoints[9] - keypoints[7]
    angle_left_elbow = get_angle(v1, v2)

    #计算左大腿和左臂夹角
    v1 = keypoints[13] - keypoints[11]
    v2 = keypoints[7] - keypoints[5]
    angle_left_leg = get_angle(v1, v2)

    #计算右大腿和右臂夹角
    v1 = keypoints[14] - keypoints[12]
    v2 = keypoints[8] - keypoints[6]
    angle_right_leg = get_angle(v1, v2)

    #计算左大腿和左小腿夹角
    v1 = keypoints[11] - keypoints[13]
    v2 = keypoints[15] - keypoints[13]
    angle_left_knee = get_angle(v1, v2)

    #推肩条件
    shoulder_push_begin= (angle_right_leg>-90 and angle_left_leg<90)
    shoulder_push_finish = (angle_right_leg<-150 and angle_left_leg>150)
    #飞鸟条件
    flying_bird_begin = (angle_right_leg>-30 and angle_left_leg<30)
    flying_bird_finish = (angle_right_leg<-60 and angle_left_leg>60)
    #深蹲条件
    squat_begin = (angle_left_knee<-120 or angle_left_knee>0)
    squat_finish = (angle_left_knee>-70 and angle_left_knee<0)
    #二头弯举条件
    bend_begin = (angle_left_elbow<180 and angle_left_elbow>150)
    bend_finish = (angle_left_elbow<45 and angle_left_elbow>0)

    if(type == "Shoulder_Push"):
        if( shoulder_push_begin):
            flag = 1
        elif( shoulder_push_finish and flag):
            Counter = 1
            flag = 0
    elif(type == "Flying_Bird"):
        if(flying_bird_begin):
            flag = 1
        elif(flying_bird_finish and flag):
            Counter  = 1
            flag = 0
    elif(type == "Squat"):
        if(squat_begin):
            flag = 1
        elif(squat_finish and flag):
            Counter = 1
            flag = 0
    elif(type == "Bend"):
        if(bend_begin):
            flag = 1
        elif(bend_finish):
            Counter = 1
            flag = 0

    return Counter

#基于TOPSIS优劣解距离的综合评价法
def get_score(keypoints, type):
    global flag
    w = [0.22, 0.21, 0.43, 0.14]
    counter1 = counter2 = counter3 = counter4 = 0

    # 直接调用train_counter获取计数
    if type == "Shoulder_Push":
        counter1 += train_counter(keypoints, type)
    elif type == "Flying_Bird":
        counter2 += train_counter(keypoints, type)
    elif type == "Squat":
        counter3 += train_counter(keypoints, type)
    elif type == "Bend":
        counter4 += train_counter(keypoints, type)
    
    # 计算small和big
    small = math.sqrt(sum(w[i] * ((counter - 5) ** 2) for i, counter in enumerate([counter1, counter2, counter3, counter4])))
    big = math.sqrt(sum(w[i] * ((20 - counter) ** 2) for i, counter in enumerate([counter1, counter2, counter3, counter4])))

    s = small / (small + big)
    score = 100 * s
    return score
