#!/usr/bin/python
# AUTHOR BeedleEiEi #
import numpy as np
import cv2 as cv
import _pickle as pickle
import matplotlib.pyplot as plt
import time
import math
import itertools


#STEP 1
""" STEP 1 Define require material :

1.1 define feature_image_database as img (model) [query]
1.2 create sift detector and adjust parameters
1.3 collect keypoint and descriptor of feature image into kpSift and model_des
1.4 define video capture as cap and read training video or cameras

"""

#1.1
img = cv.imread('feature_all_trans.png', 0)

#1.2
sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.005, edgeThreshold=30, sigma=2.0)

#1.3
kpSift,model_des = sift.detectAndCompute(img,None)

#1.4#
cap = cv.VideoCapture('{}'.format("A:/PiggySample/feature_index_database/video_feature/testVid1.mp4"))

#STEP 2
""" STEP 2 Define collection and status variables :

2.1 define keypoint and descriptor list of previous frame and present frame
2.2 define list of newly found good feature, present frame good feature and previous frame good feature
2.3 define tracking list of present_kp, present_pos, previous_kp, previous_pos, frame_timestamp
2.4 define counting variable of non-moving feature and check list
2.5 define previous image for prevent an error on start**
2.6 define status alert
2.7 define iterator for counting round, timestamp
2.8 define error collecting log (Optional)

"""
#2.1
previous_kp, previous_des = [],[]
present_kp, present_des = [],[]

#2.2 Feature collection
model_good_feature_list = []
present_good_feature_list = []
previous_good_feature_list = []

#2.3
original_and_present_kp_list = []

#2.4
same_position_feature_count = 0
check_not_moving_feature_list = []

#2.5 Define equal to original image size**
previous_image_frame = np.zeros((800,600), np.float32)

#2.6
alert = False

#2.7
iterator = 0;

#2.8
error_log = []

#STEP 3
""" STEP 3 Core function
    3.1 create FLANN matcher
    3.2 get video parameters
    3.3 use sift to detect keypoint and descriptor current frame (present_frame)
    3.4 use flann matching for previous descriptor and present descriptor (previous_frame and present_frame)
    3.5 finding good feature for previous descriptor and present descriptor
        3.5.1 thresholding feature referenced from lowe's paper: m.distance < 0.7*n.distance [the lower distance the better it is]
        3.5.2 collect feature attributes for checking moving pixel
    3.6 finding new feature in current frame
        3.6.1 Finding good feature of model thresholding feature referenced from lowe's paper: m.distance < 0.7*n.distance [the lower distance the better it is]
        3.6.2 collect feature attributes for checking moving pixel in 3.7.3
        3.6.3 Remove dupplicate feature
    3.7 Check moving feature and alert
        3.7.1 If pixel differnce is more than config it's moving
        3.7.2 If moving so far remove it..
        3.7.3 Alert and remove old feature alerted

"""
########### 3.1 CREATE FLANN #########################################
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

# Define matcher
flann = cv.FlannBasedMatcher(index_params,search_params)
######################################################################

while(cap.isOpened()):
    ###### 3.2  GET VIDEO PARAMETERS #################################
    ret, frame = cap.read()
    image_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # trainImage
    image_crop = image_frame
    image_compare_model = image_frame
    # Timestamp ของ frame
    frame_timestamp = cap.get(cv.CAP_PROP_POS_MSEC)
    # FPS
    fps = cap.get(cv.CAP_PROP_FPS)
    #################################################################

    # Prevent Error first frame
    if iterator == 0:
        previous_image_frame = image_frame

    print(' Round Iterator :', iterator)
    present_kp, present_des = sift.detectAndCompute(image_crop, None)

    ########### 3.4 MATCHING PREVIOUS AND CURRENT FRAME ############
    if iterator != 0:
        # previous_des now is model_des from previous frame
        present_matches = flann.knnMatch(previous_des,present_des, k=2)

        # sort for fast search
        present_matches = sorted(present_matches, key=lambda x:x[0].distance)

    else:
        present_matches = []


    # 3.5 Finding good feature for match model and present frame ########

    """OUTPUT FOR THIS STEP: Filtered feature -> [[DMatch1], [DMatch2]]
       If want <Keypoint> use: present_kp[DMatch.queryIdx]
       If want <Keypoint> pt use: present_kp[DMatch.queryIdx].pt
    """
    try:
        for i, (present,previous) in enumerate(present_matches):
            ########### 3.5.1 Filter thresholding ###########
            # Filter the lower queryIdx.distance the better it is [distance between descriptor]
            # m is present feature, n is previous feature
            if present.distance < 0.7*previous.distance:
                present_good_feature_list.append([present])

                """original_and_present_kp_list contain : present_kp,present_pos,previous_kp,previous_kp_pos,frame_timestamp"""

                # ต้องใช้ trainIdx จึงจะได้ ตำแหน่ง feature บนตัวหมูใน frame ไม่ใช่ที่ model!! |  a = present_kp[present.trainIdx].pt

                tmp_present_kp_position = present_kp[present.trainIdx].pt

                # get previous keypoint and position
                tmp_previous_kp = previous_kp[previous.trainIdx]
                tmp_previous_kp_position = previous_kp[previous.trainIdx].pt

                # 3.5.2 collect feature attributes for checking moving pixel ###########
                original_and_present_kp_list.append([tmp_present_kp, tmp_present_kp_position,
                                                     tmp_previous_kp, tmp_previous_kp_position,
                                                     iterator])
    except Exception as e:
        print('Error in checking good feature first match', e)

    #   3.6 finding new feature in current frame
    """ Matches Model_des,present_des เพื่อหา feature ใหม่ๆ  [ใช้ present_des,Model_des ไม่ได้เนื่องจากบางประการ]"""
    model_matches = flann.knnMatch(model_des, present_des, k=2)

    # sort for fast search
    model_matches = sorted(model_matches, key=lambda x:x[0].distance)

    # 3.6.1 Finding good feature of model
    try:
        for i, (m,n) in enumerate(model_matches):
            # เก็บ present feature ที่ดี
            if m.distance < 0.7*n.distance:
                # Add good feature to newly found feature list (model_good_feature_list)
                model_good_feature_list.append([m])

                """original_and_present_kp_list contain : present_kp,present_pos,previous_kp,previous_pos,frame_timestamp"""

                # get present keypoint and position
                tmp_present_kp = present_kp[m.trainIdx]
                tmp_present_kp_position = present_kp[m.trainIdx].pt

                # get previous keypoint and position
                # เนื่องจากเป็น feature ใหม่ จึงเก็บค่าใหม่หมด

                # 3.6.2 collect feature attributes for checking moving pixel
                original_and_present_kp_list.append([tmp_present_kp, tmp_present_kp_position,
                                                     tmp_present_kp, tmp_present_kp_position,
                                                     iterator])

                # Add good feature to present frame feature list (compared to previous frame)
                present_good_feature_list.append([m])

                # 3.6.3 Remove dupplicate feature
                # Remove dupplicate value for sure!!!
                try:
                    model_good_feature_list.sort(key=lambda x:x[0].distance)
                    present_good_feature_list.sort(key=lambda x:x[0].distance)
                    original_and_present_kp_list.sort(key=lambda x:x[4])

                    model_good_feature_list = list(model_good_feature_list for model_good_feature_list,_ in itertools.groupby(model_good_feature_list))
                    present_good_feature_list = list(present_good_feature_list for present_good_feature_list,_ in itertools.groupby(present_good_feature_list))
                    original_and_present_kp_list = list(original_and_present_kp_list for original_and_present_kp_list,_ in itertools.groupby(original_and_present_kp_list))

                except Exception as e:
                    print('Error here sort', e)
                    error_log.append('Error here sort' + str(e))

    except Exception as e:
        print('Error in checking good model feature second', e)
        error_log.append('Error in checking good model feature second' + str(e))

    # 3.7 Check moving feature and alert
    try:
        frame_time_ = ""
        for index, (present_kp_, present_pos, previous_kp_, previous_pos, og_frame_timestamp) in enumerate(original_and_present_kp_list):
            position_difference = (abs(present_pos[0] - previous_pos[0]), abs(present_pos[1] - previous_pos[1]))
            # 3.7.1 If pixel differnce is more than config it's moving
            # เราลบ feature ที่ขยับเว่อร์ออกไปได้ไหม
            if abs(position_difference[0]) > 5 or abs(position_difference[1]) > 5:
                original_and_present_kp_list[index][2] = present_kp_
                original_and_present_kp_list[index][3] = present_pos
                original_and_present_kp_list[index][4] = iterator

            # 3.7.2 If moving so far remove it..
            if abs(position_difference[0]) > 100 or abs(position_difference[1]) > 100:
                # Remove alerted feature
                original_and_present_kp_list.pop(index)

            # 3.7.3 Alert and remove old feature alerted
            else:
                frame_time_ = iterator
                """ Got frame difference in minute (diff)/fps/60 -> minute """
                if (abs(frame_time_ - og_frame_timestamp) / fps) / 60 >= 3.0:
                    same_position_feature_count += 1
                    check_not_moving_feature_list.append( tuple( (int(present_pos[0]), int(present_pos[1])) ) )

                    # Remove alerted feature
                    original_and_present_kp_list.pop(index)
                    alert = True

    except Exception as e:
        print('Error in pos calculattion', e)
        error_log.append('Error in pos calculattion' + str(e))


    #-------------------------STEP 4 SHOW OUTPUT AND CLEARING RESULT---------------------------------#
""" STEP 4 SHOW OUTPUT AND CLEARING RESULT
    Show matching result and also save snapshot then clear value} counting variable
    4.1 Copy image frame to use
    4.2 Show newly found feature in present frame
    4.3 Show feature from previous frame
    4.4 Set previous image to present
    4.5 Save snapshot (Optional)
    4.6 Alert if same_position_feature_count is more than config and save snapshot
    4.7 (Optional) Save image for keypoint found
    4.8 Clearing value for next frame

"""
    # 4.1 Copy image frame to use
    compared_feature_image = np.array(image_frame)
    new_feature_detected_image = np.array(image_compare_model)

    """Show matches via feature list"""

    # 4.2 Show newly found feature in present frame
    try:
        new_feature_detected_image = cv.drawMatchesKnn(img, kpSift, image_crop, present_kp, model_good_feature_list, image_compare_model,
                                                       singlePointColor = (255,0,0), matchColor = (0,255,0), flags=2)
        cv.imshow("New feature detected", new_feature_detected_image)

    except Exception as e:
        print('error here model image', e)
        error_log.append('error here model image' + str(e))


    if present_good_feature_list == []:
        print(' present_good_feature_list = null but nothing to worry')

    # 4.3 Show feature from previous frame
    try:
        compared_feature_image = cv.drawMatchesKnn(img, kpSift, image_crop, present_kp, present_good_feature_list, image_frame,
                                                   singlePointColor = (255,0,0), matchColor = (0,255,255), flags=2)
        cv.imshow("Tracking previous feature", compared_feature_image)

    except Exception as e:
        print('error here', e)
        error_log.append('error here' + str(e))

    # 4.4 Set previous image to present
    previous_image_frame = image_frame

    # 4.5 Save snapshot (Optional)
    # Save image snapshot
    #cv.imwrite("./newFeature{}.png".format(iterator), new_feature_detected_image)
    #cv.imwrite("./result{}.png".format(iterator), compared_feature_image)


    # 4.6 Alert if same_position_feature_count is more than config and save snapshot
    if alert and same_position_feature_count >= 10:
        print("Pig might sleeping outside!! Counted feature = ", same_position_feature_count)
        cv.imwrite("./res/newFeature{}.png".format(iterator), new_feature_detected_image)
        cv.imwrite("./res/result{}.png".format(iterator), compared_feature_image)

        alert = False
        same_position_feature_count = 0

        # 4.7 (Optional) Save image for keypoint found
        try:
            check_kp_image = frame
            check_not_moving_feature_list.sort()
            check_not_moving_feature_list = list(check_not_moving_feature_list for check_not_moving_feature_list,_ in itertools.groupby(check_not_moving_feature_list))
            for pt in check_not_moving_feature_list:
                cv.circle(check_kp_image, pt, 3, (0,0,255), -1)
                #cv.imshow("Circled check KP", check_kp_image)
                cv.imwrite("./kp/resultCheckKP{}.png".format(iterator), check_kp_image)
            check_not_moving_feature_list = []

        except Exception as e:
            print('Error in print keypoint image', e)
            error_log.append('Error in print keypoint image' + str(e))

    # 4.8 Clearing value for next frame
    try:
        # ผ่านไป 1700++ frame แล้วจะคำนวณช้าเพราะ feature เยอะเกิน ใช้ cost เยอะ
        # เก็บไว้ใน temp_kp เพื่ออัพเดทในเฟรมถัดไป (เก็บไว้ใน previous_kp, previous_des)
        previous_good_feature_list = present_good_feature_list.copy()
        present_good_feature_list = []

        model_good_feature_list = []
        previous_kp = list(present_kp).copy()
        previous_des = model_des.copy()


    except Exception as e:
        print('Error in passing value to tmp', e)
        error_log.append('Error in passing value to tmp' + str(e))


    # Press q to exit
    if cv.waitKey(1) & 0xFF == ord('c'):
        if error_log != []:
            print(error_log)
        break

    iterator += 1

cap.release()
cv.destroyAllWindows()

