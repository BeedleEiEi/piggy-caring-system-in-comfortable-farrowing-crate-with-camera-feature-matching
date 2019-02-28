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
IMAGE_FORMAT_LIST = ["png", "jpg", "bmp"]
# ============================================================================

class PolygonDrawer(object):
    def __init__(self, canva):
        self.window_name = canva[0] # Window's name
        self.canvas_size = canva[1] # Canvas size [Feature image size]
        print(self.canvas_size, "  Canvas size")
        self.feature_img = canva[2] # Feature image to be drawed

        self.done = False # Flag signalling we're done clicked
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Done
            return

        if event == cv.EVENT_MOUSEMOVE:
            # Update current mouse position
            self.current = (x, y)

        elif event == cv.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))

        elif event == cv.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        print("Canvas size : ", self.canvas_size)

        cv.namedWindow(self.window_name)

        # Show feature image
##        cv.imshow(self.window_name, np.zeros(canvas_size, np.uint8))
##        cv.waitKey(1)

        cv.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window

            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv.polylines(self.feature_img, np.array([self.points]), False, (0,0,0), 1)

                # Show what the current line would look like
##                cv.line(feature_img, self.points[-1], self.current, WORKING_LINE_COLOR)

          # Update the window
            cv.imshow(self.window_name, self.feature_img)

            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv.waitKey(50) == 27: # Detect right clicked [ESC hit]
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        canvas = np.zeros(self.canvas_size, np.uint8)

        # Fill polygon on canvas
        if (len(self.points) > 0):
            cv.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)

        # Show it
        cv.imshow(self.window_name, canvas)

        # Waiting for the user to press any key
        cv.waitKey()

        cv.destroyWindow(self.window_name)
        return canvas

# ============================================================================

class FeatureCreator(object):
    """This class contain function to create feature image"""

    def __init__(self):
        # Assign default feature size
        self.raw_feature_image = np.zeros(canvas_size, np.uint8)
        self.feature_image_size = (320,480)
        # Init feature image
        self.feature_image = np.zeros(canvas_size, np.uint8)
        self.polygon_drawer = PolygonDrawer(["PolygonDrawer",self.feature_image_size, self.feature_img])
        self.masked_feature_image = None
        self.bitwisedAnd_feature_image = None
        self.write_feature_image_location = ""

    def __init__(self, feature_image):
        """Expected feature image object from cv.imread(path directory)"""
        # We need 2 important thing : 1.Feature image, 2.Feature image size
        # Assign default feature size
        self.raw_feature_image = feature_image
        # Init feature image with gray scale
        self.feature_image = cv.cvtColor(self.raw_feature_image, cv.COLOR_BGR2GRAY)
        # Assign feature image size as tuple of height, width
        self.feature_image_size = (self.feature_image.shape[0], feature_image.shape[1])
        self.polygon_drawer = PolygonDrawer(["PolygonDrawer",self.feature_image_size, self.feature_image])
        self.masked_feature_image = None
        self.bitwisedAnd_feature_image = None
        self.write_feature_image_path = ""

    def get_size_from_image(image):
        """Return Image size as tuple (height, width)"""

        # Get image height, width
        height, width = image.shape[:2]
        img_size = (height, width)

        # Show feature image that you want to get size
##        print("feature image that you want to get size [Press any key to exit]: ", image)
##        cv.imshow("Input Image", image)
##        cv.waitKey(0)
##        cv.destroyAllWindows()

        return img_size


    def get_feature_image_size(self):
        """Return feature image size as tuple (height, width)"""
        return self.feature_image_size

    def get_feature_image(self):
        """Return feature image"""
        return self.feature_image

    def set_raw_feature_image(self, raw_feature_image):
        """Set feature image, Expect: raw image (color image)"""
        self.raw_feature_image = raw_feature_image

    def set_feature_image(self, feature_image):
        """Set feature image, Expect: raw image (color image), But gray is acceptable."""
        self.feature_image = cv.cvtColor(feature_image, cv.COLOR_BGR2GRAY)

    # ถ้า mask feature ใหม่เลยก็ใช้ set, get_masked_feature()
    def set_mask_feature(self):
        """Set mask feature image"""
        # Set mask feature by drawing polygon on feature image [Use PolygonDrawer's instance]
        self.masked_feature_image = self.polygon_drawer.run()

    def get_masked_feature(self):
        """Return masked feature image"""
        return self.masked_feature_image

    def set_bitwisedAnd_feature_image(self):
        """Set bitwise AND to feature image"""
        # Create bitwise AND then overwrite to feature image
        # Expect to get piggy with black background
        self.bitwisedAnd_feature_image = cv.bitwise_and(self.feature_image, self.masked_feature_image, self.masked_feature_image, mask=None)

    def get_bitwisedAnd_feature_image(self):
        """Get bitwise AND to feature image, Expect to get piggy with black Background"""
        return self.bitwisedAnd_feature_image

    def set_write_feature_image_path(self, path):
        """Set feature image write output path"""
        self.write_feature_image_path = path

    def get_write_feature_image_path(self):
        """Return feature image write output path"""
        return self.write_feature_image_path

    def write_bitwisedAnd_feature_image_with_timestamp(self):
        """Write bitwisedAnd feature image with timestamp (integer) to external directory"""
        # Create timestamp
        timestamp = time.time()
        # Write bitwisedAnd feature image with timestamp (integer)
        cv.imwrite("{0}/feature_bitwise_{1}.png".format(self.write_feature_image_path, timestamp), self.bitwisedAnd_feature_image)
        print("'{0}/feature_bitwise_{1}.png'".format(self.write_feature_image_path, timestamp))
        cv.imshow("Bitwised feature image", self.bitwisedAnd_feature_image)
        # Press any key to continue...
        print("Press any key to continue...")
        cv.waitKey(1)
        cv.destroyAllWindows()

#--------------------------------------------------------------------------------------------------#
def init_bgs(image):
    """Initial Background Subtraction"""
    blur = cv.GaussianBlur(image,(5,5),0)
    image = cv.addWeighted(blur,1.5,image,-0.5,0)

    # Apply BGS
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    fgbg = cv.bgsegm.createBackgroundSubtractorGMG()

def resize_feature_image_to_local_feature_size(high_resolution_feature_image):
    """This function return resized feature image\
       Expect hi resolution feature image,\
       To be resized equal to size of local video feature image for higher accuracy\
       when used hi resolution feature compared with low quality video
    """
    # We need one feature image from video to compare size with high resolution image
    feature_image_of_video = cv.imread("A:/PiggySample/feature_index_database/feature1.png", 1)
    feature_height, feature_width = feature_image_of_video.shape[:2]
    image_resized = cv.resize(high_resolution_feature_image, (feature_height, feature_width), interpolation = cv.INTER_AREA)
    return image_resized

def start_sift_matching(feature_img, video_location, save_location, draw_feature=False):
    """SIFT matching with video"""
    # Initiate SIFT detector
    # SIFT create parameter (nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04,edgeThreshold=10,sigma=1.6)
    sift = cv.xfeatures2d.SIFT_create(contrastThreshold=0.005, edgeThreshold=30, sigma=2.0)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(feature_img,None)
    #np.savetxt('feature_desc.txt', des1)

    # If want to create more than one feature
    #kp3, des3 = sift.detectAndCompute(img3, None)

    # If have more than one feature image can add into list of image set
    feature_image_list = [[feature_img,kp1,des1]]

    # Read video
    print('{}'.format(video_location))
    cap = cv.VideoCapture('{}'.format(video_location))

    # First we need to define area of sift detector, In fact we normally detect all pixel in frame,\
    # But sometimes we might used to detect on some interested region.
    if cap.isOpened():
        ret, frame = cap.read()
        # Convert ROI image into gray scale
        imgROI = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        roi = cv.selectROI(imgROI)

        # Crop image
        image_crop = imgROI[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

        # Display cropped image
        #cv.imshow("ImageCrop", image_crop)

        print("ROI size = {}".format(roi))
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Initial time to capture behavior of piggy which sleeping
    time = 0
    # Initial number of detected sleeping piggy
    sleep_count_following_rule = 0
    # Initial number of frame from video, It use to create snapshot frame number
    count_image_snapshot = 0

    # This step is mainly detect feature and matching sift
    while(cap.isOpened()):
        ret, frame = cap.read()
        image_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # trainImage

        #BGS
        #blurImg = cv.blur(image_frame,(5,5))
        #fgmask = fgbg.apply(blurImg)
        #fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

        # Create cropped image from ROI
        image_crop = image_frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

        # Start detect sift and compute
        kp2, des2 = sift.detectAndCompute(image_crop,None)

        # In case want to import descriptor (des)
        #des = np.loadtxt('descriptor_db.txt', dtype=np.float32)

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

            # No need to sort...
            # Sort match result maybe faster to detect good feature
            # But not sure it probably result in millisec so adjust key is possible choice
            # The lower the better it is.
            #matches = sorted(matches, key=lambda m:m[0].distance, reverse=True)

            # Finding good feature
            # Ratio test as per Lowe's paper
            # Try adjust threshold in this case video has low quality, It's score has less than normal adjust ratio to more than 0.7 is better detection

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


            # In case want to draw square at feature detection but now it's bug.
            # Need adjust time of sleeping behavior

            # Check good_feature_list may be obsolete last changed | Use matchesMask instead
            if draw_feature == True:
                if len(good_feature_list) > MIN_MATCH_COUNT:
                    # Increase time detec feature that sleeping
                    time += 0.1
                    if time >= 200:
                        #print(time)
                        print('Alert !!!! Piggy is sleeping outside Heatpad!!!!')
                        time = 0
                        sleep_count_following_rule += 1

                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_feature_list]).reshape(-1,1,2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_feature_list]).reshape(-1,1,2)

                    M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

                    h, w, _ = raw_feature_image.shape
                    pts = np.float32([[0 , 0],
                                      [0, h - 1],
                                      [w - 1, h - 1],
                                      [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv.perspectiveTransform(pts, M)

                    # This block draw poly lines over the output image so carefully check
                    image_draw_feature_polylines = cv.polylines(image_crop, [np.int32(dst)],
                                     True, 255, 3, cv.LINE_AA)

            # New code define draw parameter and use matchesMask instead of good_feature_image
            draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

            # To use newer version : cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,outputImage,**draw_params)
            # Check good_feature_list may be obsoleted last changed | Use matchesMask instead
##            correspondences_image = cv.drawMatches(image_i, kpi, image_crop, kp2,
##                                           good_feature_image, image_frame, flags=2)

            # Better match than good feature old code
            # Draw the keypoint matches with original image
            correspondences_image = cv.drawMatchesKnn(image_i, kpi, image_crop, kp2, matches, image_frame, **draw_params)

            # Show square feature detected
            #cv.imshow("Detect sleeping piggy", image_draw_feature_polylines)

            # Show original frame
            #cv.imshow("Original frame", image_frame)

            #Show matched lines with original Image
            cv.imshow("Correspondences", correspondences_image)

            # Show Background Subtraction of video
            # cv.imshow(str(image_i), image_i)
            # cv.imshow("BGS", fgmask)

            if save_location != False:
                cv.imwrite("{0}{1}.jpg".format(save_location, count_image_snapshot), correspondences_image)
                count_image_snapshot += 1

        # Press q to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

#------------------------------------------------------------------------------------------------------------------------------#
def start(feature_bitwise, video_location, save_location, draw_feature=False):
    try:
        start_sift_matching(feature_bitwise, video_location, save_location, draw_feature)
    except Exception as e:
        print(e)
        print(video_location)
        print("Out of frame")
        print("Your snapshot has been saved at {}".format(save_location))
        exit(1)

def check_image_format(image_path):
    """"""
    try:
        is_image_format = image_path[-3:] in IMAGE_FORMAT_LIST
        return is_image_format
    except Exception as e:
        print(e, '    File format is incorrect')

def create_new_feature_by_image(image_path, save_feature_image_path, use_high_resolution_feature_image=False):
    """Create new feature"""
    # Step 1 : We need at least one image to initial feature, So define path of your feature image
    # If use image from video use this block
    try:
        image_path = '{}'.format(image_path)
        if check_image_format(image_path) == True:
            image = cv.imread(image_path)
        else:
            # It might be video format
            feature_image_cap = cv.VideoCapture(image_path)
            if feature_image_cap.isOpened():
                ret, frame = feature_image_cap.read()
                imgROI = frame
                roi = cv.selectROI(imgROI)

                # Crop image
                image = imgROI[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

                # Display cropped image
                cv.imshow("ROI OF Feature Image From Video", image)

                print("ROI Feature Image From Video Size = {}".format(roi))
                print("Press any key to continue bitwise...")
                cv.waitKey(0)
                cv.destroyAllWindows()

        # If feature image is high resolution image
        if use_high_resolution_feature_image == True:
            image = resize_feature_image_to_local_feature_size(image)

        # Step 2 : Create feature creator instance to create feature from image
        # AUTHOR NOTE : This script expect to get bitwised(AND) of a feature image with black background in it.
        feature_creator = FeatureCreator(image)

        # Step 3 : Draw polygon to mask feature image
        feature_creator.set_mask_feature()

        # Step 4 : Create bitwised(AND) of feature image by passing masked image
        feature_creator.set_bitwisedAnd_feature_image()

        # Step 5 : Set path to write bitwised image
        feature_creator.set_write_feature_image_path(save_feature_image_path)

        # Step 6 : Save feature image with black background
        feature_creator.write_bitwisedAnd_feature_image_with_timestamp()

    except Exception as e:
        print(e)

if __name__ == "__main__":
                                                        ## MAIN STATE ##

    #-------------------------IN CASE WE WANT TO CREATE NEW FEATURE IMAGE WITH BLACK BACKGROUND IN IT--------------------------#

    #----------------------------------------SKIP IF ALREADY HAVE FEATURE IMAGE------------------------------------------------#

    create_new_feature = False
    use_high_resolution_feature_image = False

    # Step 1 : Load feature image
    video_location = "A:/PiggySample/feature_index_database/video_feature/feature1.mp4"

    if create_new_feature == True:
        # Save created feature image
        image_to_create_feature_path = "A:/PiggySample/feature_index_database/video_feature/feature5.mp4"
        save_feature_image_path = "A:/PiggySample/feature_index_database/masked_feature2/"
        create_new_feature_by_image(image_to_create_feature_path, save_feature_image_path, use_high_resolution_feature_image)

    # Save snapshot from feature matching
    save_location = "A:/PiggySample/update_sift_create/result/ratio 0.80/front-camera/new-feature1/"
    feature_bitwise = cv.imread("A:/PiggySample/feature_index_database/masked_feature2/feature_bitwise_1.png")
    # Step 2 : Start feature matching by passing feature image into start_sift_matching()
    start(feature_bitwise, video_location, save_location, draw_feature=False)

#------------------------------------------------------------------------------------------------------------------------------#
