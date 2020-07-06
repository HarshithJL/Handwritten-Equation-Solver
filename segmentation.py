import numpy as np
import cv2
import operator


MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 45
RESIZED_IMAGE_HEIGHT = 45

class ContourWithData():
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA: return False
        return True


allContoursWithData = []
validContoursWithData = []
img = cv2.imread('quadratic.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlurred = cv2.GaussianBlur(gray, (1,1), 0)
ret,imgThresh = cv2.threshold(imgBlurred,95,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
npaContours,npaHierarchy = cv2.findContours(imgThresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

imgThresh2= cv2.bitwise_not(imgThresh)
for npaContour in npaContours:
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)

for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list


validContoursWithData.sort(key = operator.attrgetter("intRectX"))
n= len(validContoursWithData)       # sort contours from left to right
del validContoursWithData[n-3:]

test_data = []

for contourWithData in validContoursWithData:            # for each contour
        imgROI = imgThresh2[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight+5,
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth+5]

        im_resize = cv2.resize(imgROI,(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT),interpolation=cv2.INTER_AREA)
        cv2.imshow("Output",im_resize)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        test_data.append(im_resize)
