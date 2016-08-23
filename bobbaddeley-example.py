import cv2
import cv2.cv as cv
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from time import time
from os.path import isfile, join
mypath = "6b-router-color"
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

drag = False
drag_start = (0,0)
drag_end = (0,0)
patterns = []
regions = []
show_regions = False
show_mask = True

def on_mouse(event, x, y, flags, params):
	global drag, drag_start, drag_end, img, patterns, regions

	if event == cv2.EVENT_LBUTTONDOWN:
		drag_start = (x, y)
		drag_end = (x, y)
		drag = True

	elif event == cv2.EVENT_LBUTTONUP:
		drag = False
		drag_end = (x, y)
		if (drag_end[1]-drag_start[1])>10 and (drag_end[0]-drag_start[0])>10:
			crop = img[drag_start[1]:drag_end[1],drag_start[0]:drag_end[0]]
			name = (str)(drag_start[0])+'-'+(str)(drag_start[1])+'.jpg'
			cv2.imwrite(mypath+'/'+name,crop)
			pattern = cv2.imread(mypath+"/"+name,1)
			patterns.append(cv2.cvtColor(pattern,cv2.COLOR_RGB2RGBA))
			pre = (name.split('.',1))[0]
			rectparts = pre.split('-')
			h,w = pattern.shape[0:2]
			rectparts.append(w)
			rectparts.append(h)
			regions.append(rectparts)
			drag_start = (0,0)
			drag_end = (0,0)
	elif event == cv2.EVENT_MOUSEMOVE and drag==True:
		drag_end = (x,y)

def show_webcam():
	global drag_start, drag_end, img, patterns, regions, show_regions, show_mask
	zoom = False
	cam = cv2.VideoCapture(0)
	cam.set(cv.CV_CAP_PROP_FRAME_WIDTH,1280)
	cam.set(cv.CV_CAP_PROP_FRAME_HEIGHT,1024)
	ret_val, img = cam.read()
	mask = cv2.imread("mask.png",0)
	cv2.namedWindow('Quietyme AOI')
	cv2.setMouseCallback('Quietyme AOI', on_mouse, 0)
	for i in onlyfiles:
		pattern = cv2.imread(mypath+"/"+i,1)
		patterns.append(cv2.cvtColor(pattern,cv2.COLOR_RGB2RGBA))
		pre = (i.split('.',1))[0]
		rectparts = pre.split('-')
		h,w = pattern.shape[0:2]
		rectparts.append(w)
		rectparts.append(h)
		regions.append(rectparts)
	while True:
		ret_val, img = cam.read()
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		char = chr(cv2.waitKey(1) & 255)
		if (char == 'q'):
			break	
		elif (char == 27):
			break
		elif (char == 't'):
			cv2.imwrite("output.jpg", img)
		elif (char == 'd'):
			show_regions = not show_regions
		elif (char == 's'):
			show_mask = not show_mask
		elif (char =='z'):
			zoom = not zoom
		#diff = cv2.absdiff(ref_image,img)
		img = cv2.cvtColor(img,cv2.COLOR_RGB2RGBA)
		if (show_mask):
			diff = cv2.bitwise_and(img,img,mask=mask)
			for i in range(len(patterns)):
				sub = img[(int)(regions[i][1])-10:(int)(regions[i][1])+(int)(regions[i][3])+10,(int)(regions[i][0])-10:(int)(regions[i][0])+(int)(regions[i][2])+10]
				if (show_regions==True):
					cv2.rectangle(diff,((int)(regions[i][0])-10,(int)(regions[i][1])-10), ((int)(regions[i][2])+(int)(regions[i][0])+10,(int)(regions[i][3])+(int)(regions[i][1])+10), (0,255,0), 1)
				res = cv2.matchTemplate(sub,patterns[i],cv2.TM_CCOEFF_NORMED)
				h,w = patterns[i].shape[0:2]
				threshold = 0.80
				loc = np.where (res >= threshold)
				for pt in zip(*loc[::-1]):
					cv2.rectangle(diff,(pt[0]+(int)(regions[i][0])-10,pt[1]+(int)(regions[i][1])-10), (pt[0] + (int)(regions[i][0])+w-10, pt[1]+(int)(regions[i][1])+h-10), (0,0,0), -1)
					cv2.rectangle(diff,drag_start, drag_end, (255,0,0), 1)
			if (zoom):
				sub = img[320:454,544:680]
				resized_image = cv2.resize(sub, (0,0),fx=4,fy=4)
				diff[119:655,340:884] = resized_image
		else:
			diff = img
		cv2.imshow('Quietyme AOI', diff)
	cv2.destroyAllWindows()

def main():
	show_webcam()

if __name__ == '__main__':
	main()
