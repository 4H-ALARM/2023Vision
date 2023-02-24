import cv2
import numpy as np

# global variables go here:
redbalence = 2050
bluebalence = 1000
testVar = 0
lowerCone = [18,0,175]
upperCone = [30,255,255]
lowerCube = [140, 40, 30]
upperCube = [180, 255, 255]
cubeAreaMinMax =[100, 250000]
cubeRadiusMinMax = [10, 50]

def convex_hull_pointing_up(ch):
    '''function to check if the polygon represents a cone pointing up'''

    # Check the points above and below the center
    points_above_center, points_below_center = [], []

    x, y, w, h = cv2.boundingRect(ch)
    aspect_ratio = w / h  # aspect ratio

    # if width is smaller than height, than polygon is a cone
    if aspect_ratio < 0.8:
        # find the center of the cone
        vertical_center = y + h / 2

        for point in ch:
            if point[0][1] < vertical_center:  # checks if the y coordinate of the point is less than verticle center, than point is above
                points_above_center.append(point)
            elif point[0][1] >= vertical_center:
                points_below_center.append(point)

        # find the leftmost and rightmost x coordinate of the point
        left_x = points_below_center[0][0][0]
        right_x = points_below_center[0][0][0]
        for point in points_below_center:
            if point[0][0] < left_x:
                left_x = point[0][0]
            if point[0][0] > right_x:
                right_x = point[0][0]

        # if points above the center lie outside the base, than it is not a cone
        for point in points_above_center:
            if (point[0][0] < left_x) or (point[0][0] > right_x):
                return False
    else:
        return False

    return True

def getRotatedRects (img,bounding_rects,angle):
    center = (img.shape[0]//2,img.shape[1]//2)
    rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_rects = []
    for rect in bounding_rects:
        x,y,w,h = rect
        bb = np.array(((x, y), (x+w, y), (x+w, y+h), (x, y+h)))
        bb_rotated = np.vstack((bb.T, np.array((1, 1, 1, 1))))
        bb_rotated = np.dot(rotMat, bb_rotated).T
        #print(bb_rotated)
        p2, p4, p3, p1 = bb_rotated
        # print(f"{p1}, {p2}, {p3}, {p4}")
        width = int(p2[0] - p1[0])
        height = int(p3[1] - p1[1])
        x = p1[0] + (width // 2)
        y = p1[1] + (height // 2)
        rotated_rect = int(x), int(y), width, height
        rotated_rects.append(rotated_rect)
    return rotated_rects

def getCones (imgHsv, img, angle):
    lower = np.array(lowerCone)
    upper = np.array(upperCone)
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask = mask)
    kernel = np.ones((5, 5))
    img_thresh_opened = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    #imgBlur = cv2.GaussianBlur(result, (7,7), 1)
    image = cv2.GaussianBlur(img_thresh_opened,(7,7), 1)

    imgcopy = image.copy()
    if angle == 90:
        imgcopy = cv2.rotate(imgcopy, cv2.ROTATE_90_CLOCKWISE)
    elif angle == -90:
        imgcopy = cv2.rotate(imgcopy, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_edges = cv2.Canny(imgcopy, 80, 160)
    contours, hierarchy = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img_contours = np.zeros_like(img_edges)
    # cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 2)
    approx_contours = []

    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed=True)
        approx_contours.append(approx)

    # img_approx_contours = np.zeros_like(img_edges)
    # cv2.drawContours(img_approx_contours, approx_contours, -1, (255, 255, 255), 1)

    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv2.convexHull(ac))

    # img_all_convex_hulls = np.zeros_like(img_edges)
    # cv2.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255, 255, 255), 2)

    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 5 <= len(ch) <= 10:
            convex_hulls_3to10.append(cv2.convexHull(ch))

    img_convex_hulls_3to10 = np.zeros_like(img_edges)
    cv2.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255, 255, 255), 2)

    cones = []
    bounding_rects = []
    for ch in convex_hulls_3to10:
        if convex_hull_pointing_up(ch):
            rect = cv2.boundingRect(ch)
            x, y, w, h = rect
            if w > 0 and h > 0:
                cones.append(ch)
                bounding_rects.append(rect)
    if angle != 0:
        bounding_rects = getRotatedRects(imgcopy,bounding_rects,angle)
    return cones, bounding_rects, image

def getCubes(imgHsv, img):
    lower = np.array(lowerCube)
    upper = np.array(upperCube)
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    kernel = np.ones((5, 5))
    img_thresh_opened = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    # imgBlur = cv2.GaussianBlur(result, (7,7), 1)
    img_thresh_blurred = cv2.GaussianBlur(img_thresh_opened, (7, 7), 1)
    img_edges = cv2.Canny(img_thresh_blurred, 80, 160)
    contours, hierarchy = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(imgCpy, contours, -1, (255, 255, 255), 2)

    cubes = []
    bounding_rects = []

    for c in contours:
        # c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        # print(area)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # print(x, y)
        # print(radius)
        if radius > cubeRadiusMinMax[0] and radius < cubeRadiusMinMax[1] and area > cubeAreaMinMax[0] and area < cubeAreaMinMax[1]:
            #cv2.circle(imgCpy, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            rect = cv2.boundingRect(c)
            cubes.append(c)
            bounding_rects.append(rect)

    return cubes, bounding_rects, img_thresh_blurred


# To change a global variable inside a function,
# re-declare it with the 'global' keyword
def incrementTestVar():
    global testVar
    testVar = testVar + 1
    if testVar == 100:
        print("test")
    if testVar >= 200:
        print("print")
        testVar = 0

def drawDecorations(image):
    cv2.putText(image, 
        'Limelight python script!', 
        (0, 230), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        .5, (0, 255, 0), 1, cv2.LINE_AA)
    
# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, llrobot):
    
    imgCpy = image.copy()
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    imgHsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    cones,bounding_rects, imgCone = getCones(imgHsv, img, 0)

     #90 cones
    cones90,bounding_rects90, _ = getCones(imgHsv, img,90)
    cones.extend(cones90)
    bounding_rects.extend(bounding_rects90)

    #-90 cones
    cones270, bounding_rects270, _ = getCones(imgHsv, img, -90)
    cones.extend(cones270)
    bounding_rects.extend(bounding_rects270)

    cubes, bounding_rect_cubes, imgCube = getCubes(imgHsv, img)

    largestContour = np.array([[]])
    llpython = [0,0,0,0,0,0,0,0]
    tcornxy = [0,0,0,0,0,0,0,0]

    largeContours = []
    json = {"Results": {"Classifier": [23,45], "Detector": [56, 78]}}

    if len(cones)>0:
        #img_cones = np.zeros_like(img_edges)
        cv2.drawContours(imgCpy, cones, -1, (255, 255, 255), 2)
        #cv2.drawContours(img_cones, bounding_rects, -1, (1, 255, 1), 2)
        for rect in bounding_rects:
            x, y, w, h = rect
            cv2.rectangle(imgCpy, (x, y), (x + w, y + h), (1, 255, 1), 3)
            center = x + w // 2, y + h // 2
            cv2.putText(imgCpy,"Cone",center,cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        contour = max(cones, key=cv2.contourArea)
        largeContours.append(contour)
        
        #x,y,w,h = cv2.boundingRect(contour)
        #llpython = [3,x,y,w,h,9,8,7]
  

    if len(cubes)>0:
        #img_cones = np.zeros_like(img_edges)
        #cv2.drawContours(imgCpy, cones, -1, (255, 255, 255), 2)
        #cv2.drawContours(img_cones, bounding_rects, -1, (1, 255, 1), 2)
        for rect in bounding_rect_cubes:
            x, y, w, h = rect
            center = x + w // 2, y + h // 2
            radius = (w + h) // 4
            #cv2.rectangle(imgCpy, (x, y), (x + w, y + h), (1, 255, 1), 3)
            cv2.circle(imgCpy, center, int(radius), (0, 255, 255), 2)
            cv2.putText(imgCpy,"Cube",center,cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        contour = max(cubes, key=cv2.contourArea)
        largeContours.append(contour)
        #x,y,w,h = cv2.boundingRect(contour)
        #tcornxy = [10,x,y,w,h,9,8,7]

    if len(largeContours) > 0:
        largestContour = max(largeContours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largestContour)
        #if len(cones) > 0 and largestContour in cones:
        #    llpython = [3,x,y,w,h,9,8,7]
        #else:
        #    llpython = [10,x,y,w,h,9,8,7]

    incrementTestVar()
    drawDecorations(imgCpy)
       
    # make sure to return a contour,
    # an image to stream,
    # and optionally an array of up to 8 values for the "llpython"
    # networktables array
    return largestContour, imgCube, llpython
    