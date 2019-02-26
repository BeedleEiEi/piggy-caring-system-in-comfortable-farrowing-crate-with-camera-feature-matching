#!/usr/bin/python
# AUTHOR BeedleEiEi #
import numpy as np
import cv2 as cv
import _pickle as pickle
import matplotlib.pyplot as plt
import time

MIN_MATCH_COUNT = 10
FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)
# ============================================================================
def start_sift_matching(feature_img, video_location, save_location, draw_feature=False):
    """SIFT matching with video"""
    # Initiate SIFT detector
    # SIFT create parameter (nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04,edgeThreshold=10,sigma=1.6)
    sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.005, edgeThreshold=30, sigma=2.0)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(feature_img,None)

    # If have more than one feature image can add into list of image set
    feature_image_list = [[feature_img,kp1,des1]]

    # This step is mainly detect feature and matching sift
    cap = cv.VideoCapture('{}'.format(video_location))
    while(cap.isOpened()):
        ret, frame = cap.read()
        image_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # trainImage

        # Create cropped image from ROI
        image_crop = image_frame

        # Start detect sift and compute
        kp2, des2 = sift.detectAndCompute(image_crop,None)

        # This step define fast library approximate nearest neighbor (FLANN) parameter to match sift feature
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        # Define matcher
        flann = cv.FlannBasedMatcher(index_params,search_params)

        # This loop start from reading feature image from feature_image_list to match each feature and sift
        # Now we have only one feature so loop is running in one feature image
        for image_i, kpi, desi in feature_image_list:

            # Start matching feature image with image frame from video
            matches = flann.knnMatch(desi,des2, k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]
            
            # Define good feature list
            # good_feature_list contain number of match
            good_feature_list = []

            # This is newer code use matchesMask to match good feature may be better than good_feature_list
            for i, (m,n) in enumerate(matches):
                if m.distance < 0.80*n.distance:
                    matchesMask[i]=[1,0]
                    good_feature_list.append(m)

            # Copy image frame to use
            correspondences_image = np.array(image_frame)
            image_draw_feature_polylines = correspondences_image

            # New code define draw parameter and use matchesMask instead of good_feature_image
            draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

            # Better match than good feature old code
            # Draw the keypoint matches with original image
            correspondences_image = cv.drawMatchesKnn(image_i, kpi, image_crop, kp2, matches, image_frame, **draw_params)

            #Show matched lines with original Image
            cv.imshow("Correspondences", correspondences_image)

            if save_location != False:
                cv.imwrite("{0}{1}.jpg".format(save_location, count_image_snapshot), correspondences_image)
                count_image_snapshot += 1

        # Press q to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

#------------------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":
                                                        ## MAIN STATE ##

#------------------------------------------------------------------------------------------------------------------------------#
    # Step 1 : Load feature image
    video_location = "A:/PiggySample/21pm_24fps.mp4"
    save_location = "A:/PiggySample/update_sift_create/result/ratio 0.80/front-camera/nigga/"
    feature_bitwise = cv.imread("A:/PiggySample/feature_index_database/masked_feature/feature_bitwise_1.png")
    # Step 2 : Start feature matching by passing feature image into start_sift_matching()
    try:
        start_sift_matching(feature_bitwise, video_location, save_location=False, draw_feature=False)
    except Exception as e:
        print(e)
        print(video_location)
        print("Out of frame")
        print("Your snapshot has been saved at {}".format(save_location))
        exit(1)
