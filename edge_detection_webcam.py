import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import numpy as np 
import cv2

def LiveCamEdgeDetection_canny(image_color):
	threshold_1 = 30
	threshold_2 = 80
	image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
	canny = cv2.Canny(image_gray, threshold_1, threshold_2)


	return canny

def LiveCamEdgeDetection_Laplacian(image_color):
	image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
	laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)

	return laplacian


def LiveCamEdgeDetection_sobely(image_color):
	image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
	y_sobel = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize = 7)

	return y_sobel


def LiveCamEdgeDetection_sobelx(image_color):
	image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
	x_sobel = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize = 7)

	return x_sobel


cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	cv2.imshow('Live Edge Detection', LiveCamEdgeDetection_canny(frame))
	cv2.imshow('Live Edge Detection Laplacian', LiveCamEdgeDetection_Laplacian(frame))
	cv2.imshow('Live Edge Detection Sobel X', LiveCamEdgeDetection_sobelx(frame))
	cv2.imshow('Live Edge Detection Sobel Y', LiveCamEdgeDetection_sobely(frame))

	cv2.imshow('Webcam video', frame)
	if cv2.waitKey(1) == 13:
		break

cap.release()
cv2.destroyAllWindows()


