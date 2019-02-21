import numpy as np
import cv2 as cv
import _pickle as pickle
import matplotlib.pyplot as plt
import time

MIN_MATCH_COUNT = 10
FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)


# ============================================================================

class PolygonDrawer(object):
    def __init__(self, canva):
        self.window_name = canva[0] # Window's name
        self.canvas_size = canva[1] # Canvas size [Feature image size]
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
        print("Canvas size : ", canvas_size)
        
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
                cv.polylines(feature_img, np.array([self.points]), False, (0,0,0), 1)
                
                # Show what the current line would look like
##                cv.line(feature_img, self.points[-1], self.current, WORKING_LINE_COLOR)
                
          # Update the window
            cv.imshow(self.window_name, feature_img)
            
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv.waitKey(50) == 27: # Detect right clicked [ESC hit]
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        canvas = np.zeros(canvas_size, np.uint8)
        
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

    def __init__(self, feature_image):
        """Expected feature image object from cv.imread(path directory)"""
            # We need 2 important thing : 1.Feature image, 2.Feature image size
            # Assign default feature size
            self.raw_feature_image = feature_image
            # Init feature image with gray scale
            self.feature_image = cv.cvtColor(raw_feature_image, cv.COLOR_BGR2GRAY)
            # Assign feature image size as tuple of height, width
            self.feature_image_size = (feature_image.shape[:1], feature_image.shape[1:2])
            self.polygon_drawer = PolygonDrawer(["PolygonDrawer",self.feature_image_size, self.feature_img])
            self.masked_feature_image = None

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
        self.masked_feature_image = self.polygon_drawer.run()

    def get_masked_feature(self, masked_feature_image):
        """Return masked feature image"""
        return self.masked_feature_image

    def set_bitwisedAnd_feature_image(self):
        """Set bitwise AND to feature image"""
        # Create bitwise AND then overwrite to feature image
        # Expect to get piggy with black background
        self.bitwisedAnd_feature_image = cv.bitwise_and(self.feature_img, self.bitwisedAnd_feature_image, self.bitwisedAnd_feature_image, mask=None)
    
    def get_bitwisedAnd_feature_image(self):
        """Get bitwise AND to feature image, Expect to get piggy with black Background"""
        return self.bitwisedAnd_feature_image

    def write_bitwisedAnd_feature_image_with_timestamp():
        """Write bitwisedAnd feature image with timestamp (integer) to external directory"""
        # Create timestamp
        timestamp = time.time()
        # Write bitwisedAnd feature image with timestamp (integer)
        cv.imwrite("./feature_index_database/masked_feature/feature_bitwise_%d.png" % timestamp, self.bitwisedAnd_feature_image)
        cv.imshow("Bitwise", self.bitwisedAnd_feature_image)
        # Press any key to continue...
        print("Press any key to continue...")
        cv.waitKey(1)
        cv.destroyAllWindows()

def testRun(feature_img):

    #blur = cv.GaussianBlur(img1,(5,5),0)
    #img1 = cv.addWeighted(blur,1.5,img1,-0.5,0)

    ##cv.imshow("Refined edge", smooth)
    ##cv.waitKey(0)
    ##cv.destroyAllWindows()

    #Apply BGS
    #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    #fgbg = cv.bgsegm.createBackgroundSubtractorGMG()

    img1 = feature_img

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp3, des3 = sift.detectAndCompute(img3, None)


    imgSet = [[img1,kp1,des1]]#, [img3,kp3,des3]]

    cap = cv.VideoCapture('A:/PiggySample/testVid1.mp4')

    if cap.isOpened():
        ret, frame = cap.read()
        imgROI = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # trainImage
        r = cv.selectROI(imgROI)



        # Crop image
        imCrop = imgROI[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        # Display cropped image
        #cv.imshow("ImageCrop", imCrop)

        print(r)
        cv.waitKey(0)
        cv.destroyAllWindows()

    time = 0
    countAppear = 0
    countImgSnapshot = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # trainImage

        #BGS
        #blurImg = cv.blur(img2,(5,5))
        #fgmask = fgbg.apply(blurImg)
        #fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

        #create cropped image from ROI
        imCrop = img2[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]


        kp2, des2 = sift.detectAndCompute(imCrop,None)


        #kp2, des2 = sift.detectAndCompute(img2,None)




        #c = np.loadtxt('descriptor_db.txt', dtype=np.float32)


        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv.FlannBasedMatcher(index_params,search_params)

        for imgi, kpi, desi in imgSet:

            matches = flann.knnMatch(desi,des2, k=2)

            #print(matches)
            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]
    ##        matches = sorted(matches, key=lambda m:m[0].distance < 0.7*m[1].distance)

            # sort match แต่ไม่รู้โอเคไหม ช่วยเร็วขึ้น หรือ แม่นขึ้นหรือไม่?
            # อาจต้องลอง sort key ที่ดีกว่า
            #matches = sorted(matches, key=lambda m:m[1].distance, reverse=True)
            #check วิธี sort ด้วยเผื่อเร็วขึ้น
            matches = sorted(matches, key=lambda m:m[0].distance, reverse=True)


            # ratio test as per Lowe's paper
            #good same as matchesMask
            #Try adjust threshold

            good = []
            for m_n in matches:
                #define m_n to check match size [match size should be 2]
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < 0.80*n.distance:
                    #matchesMask[i]=[1,0]
                    good.append(m)

            #Try to use cropped image
            img_show = np.array(img2)


            #img_show = np.array(img2)

            #Need adjust time
    ##        if len(good) > MIN_MATCH_COUNT:
    ##
    ##            #sleep(0.1)
    ##            time += 0.1
    ##            #print(time, "    ---D")
    ##            if time >= 200:
    ##                #print(time)
    ##                print('Alert !!!! Piggy is sleeping outside Heatpad!!!!')
    ##                time = 0
    ##                countAppear += 1
    ##
    ##            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    ##            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    ##
    ##            M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    ##
    ##            h, w, _ = raw_feature_image.shape
    ##            pts = np.float32([[0 , 0],
    ##                              [0, h - 1],
    ##                              [w - 1, h - 1],
    ##                              [w - 1, 0]]).reshape(-1, 1, 2)
    ##            dst = cv.perspectiveTransform(pts, M)
    ##
    ##
    ####            img2 = cv.polylines(img_show, [np.int32(dst)],
    ####                                         True, 255, 3, cv.LINE_AA)
    ##            imCrop = cv.polylines(imCrop, [np.int32(dst)],
    ##                             True, 255, 3, cv.LINE_AA)
    ##        ############################################################################


                # Draw the keypoint matches

            img_show = cv.drawMatches(imgi, kpi, imCrop, kp2,
                                           good, img2, flags=2)

    ##        show original img
    ##        img_show = cv.drawMatches(imgi, kpi, img2, kp2,
    ##                                       good, img2, flags=2)
    ##

            #cv.imshow("Correspondences", img2)

    ##        cv.imshow("OriginalImage", img2)

            #Show เส้น พร้อม Original Image
            cv.imshow("Correspondences2", img_show)

    ##        cv.imshow(str(imgi), imgi)
    ##        cv.imshow("BGS", fgmask)

            #cv.imwrite("A:/PiggySample/filter_2_feature_8.0/frame%d.jpg" % countImgSnapshot, img_show)
            countImgSnapshot += 1


        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        #showMatches(img1,kp1,img2,kp2,matches, matchesMask, imgSet)

    cap.release()
    cv.destroyAllWindows()

#----------------------------------------------------------------------------------------------#
if __name__ == "__main__":
                                                        ## MAIN STATE ##

    #-------------------------IN CASE WE WANT TO CREATE NEW FEATURE IMAGE WITH BLACK BACKGROUND IN IT-------------------------#

    #----------------------------------------SKIP IF ALREADY HAVE FEATURE IMAGE------------------------------------------------#

    # Step 1 : We need at least one image to initial feature
    image = cv.imread('A:/PiggySample/feature_index_database/feature7.png')

    # Step 2 : Create feature creator instance to create feature from image
    # AUTHOR NOTE : This script expect to get bitwised(AND) of a feature image with black background in it.
    feature_creator = FeatureCreator(image)


    

###########################################################################

    #feature_bitwise = cv.imread('A:/PiggySample/feature_index_database/masked_feature/feature_bitwise_collapse_1.png')
    #testRun(feature_bitwise)



    #วาด mask feature ทับลงไป


##    print("Polygon = %s" % pd.points)

