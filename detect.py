# import the necessary packages
import numpy as np
import argparse
import cv2
import math
import os
import glob
import configparser
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = False, help = "Path to the input image(s)")
ap.add_argument("-o", "--output", required = False, help = "Path to save the output data")
args = vars(ap.parse_args())

if args["input"] is None:
    imtypes = [("TIFF", "*tif *.tiff"),
               ("JPEG", "*.jpg *.jpeg *.jpe"),
               ("PNG", "*.png *.pns")]
    filez = filedialog.askopenfilenames(title='Select input images', filetypes=imtypes)
    files = root.tk.splitlist(filez)
else:
    files = glob.glob(os.path.join(args["input"], '*.jpeg'))

if args["output"] is None:
    ftype = [('CSV file', '*.csv')]
    filez = filedialog.asksaveasfile(filetypes=ftype, defaultextension=ftype)
    if filez is None:
        exit()
    fout = open(filez.name, filez.mode)
else:
    fout = open(args["output"], 'w')
fout.write('file; x; y; r; nucleus; cytoplasm; median\n')
fout.flush()

# load the image, clone it for output, and then convert it to grayscale
for file in sorted(files):
    image = cv2.imread(file)
    factor = min(screen_height / image.shape[0], screen_width / image.shape[1]) * 0.9
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_median = int(np.median(gray))

    config = configparser.ConfigParser()
    config.read("config.ini")
    minDist = float(config.get("detector", "minDist"))
    param1 = float(config.get("detector", "param1"))
    param2 = float(config.get("detector", "param2"))
    minRadius = int(config.get("detector", "minRadius"))
    maxRadius = int(config.get("detector", "maxRadius"))

    manual = config.get("measure", "manual") == "yes"

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    global selected_circles
    selected_circles = []

    def show_circles():
        global circles, selected_circles
        circles_ = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles_:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (255, 0, 0), 2)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        for (x, y, r) in selected_circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        output_ = cv2.resize(output, None, fx=factor, fy=factor)
        cv2.imshow("output", output_)

    def euclidean_dst(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    last_rclick = None
    def mouse_callback(event, x, y, flags, param):
        global circles, selected_circles, image, last_rclick
        output = image.copy()
        x = int(round(x / factor))
        y = int(round(y / factor))
        if event == cv2.EVENT_LBUTTONDOWN and circles.shape[1] > 0:            
            closer_circle = min(np.round(circles[0, :]).astype("int"), 
                                key=lambda c: euclidean_dst(c, (x, y)))

            if  euclidean_dst(closer_circle, (x, y)) <= closer_circle[2]:
                if tuple(closer_circle) in selected_circles:
                    selected_circles.remove(tuple(closer_circle))
                else:
                    selected_circles.append(tuple(closer_circle))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if last_rclick is None:
                last_rclick = (x, y)
            else:
                r = euclidean_dst(last_rclick, (x, y))
                x, y = last_rclick
                circles = np.hstack([circles, [[[x, y, r]]]]).astype(np.float32)
                selected_circles.append((int(x), int(y), int(r)))
                last_rclick = None

    # ensure at least some circles were found
    if circles is None:
        circles = np.empty((1,0,3))
    cv2.namedWindow('output')
    cv2.setMouseCallback('output', mouse_callback)
    # convert the (x, y) coordinates and radius of the circles to integers
    while(1):
        show_circles()
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            fout.close()
            exit()
        elif key == 13:
            break

    def find_peaks(hist, min_dst = 10, min_frec = .01):
        peaks = []
        for i in range(256):
            if hist[i] == max(hist_mask[max(0, i-min_dst):min(255, i+min_dst)]) and \
            hist[i] > sum(hist) * min_frec:
                peaks.append(i)
        return peaks

    for (x, y, r) in selected_circles:
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(mask, (x, y), r, 255, cv2.FILLED)
        hist_mask = cv2.calcHist([gray], [0], mask, [256], [0,256])
        import matplotlib.pyplot as plt
        plt.plot(hist_mask)
        peaks = find_peaks(hist_mask)
        if len(peaks) >= 2 and not manual:
            peaks = sorted(sorted(peaks, key=lambda d: hist_mask[d])[-2:])
        else:
            inset = gray[max(0, y-r):min(y+r, gray.shape[0] - 1), 
                        max(0, x-r):min(x+r, gray.shape[1] - 1)]

            global selected_peaks
            selected_peaks = []
            s = 5
            def select_callback(event, x, y, flags, param):
                global selected_peaks
                if event == cv2.EVENT_LBUTTONDOWN:
                    peak = np.median(inset[max(0, y-s):min(y+s, inset.shape[0] - 1), 
                                        max(0, x-s):min(x+s, inset.shape[1] - 1)])
                    if len(selected_peaks) < 2:
                        selected_peaks.append((x, y, peak))
                        
            cv2.namedWindow('select')
            cv2.setMouseCallback('select', select_callback)
            while(1):
                output = inset.copy()
                for (x, y, _) in selected_peaks:
                    cv2.rectangle(output, (x - s, y - s), (x + s, y + s), (0, 128, 255), -1)
                cv2.imshow('select', output)
                key = cv2.waitKey(20) & 0xFF
                if key == 27:
                    fout.close()
                    exit()
                elif key == 13:
                    if len(selected_peaks) == 2:
                        peaks = sorted([p[2] for p in selected_peaks])
                        cv2.destroyWindow('select')
                        break
                elif key == 8:
                    selected_peaks.pop()
        csv = (x, y, r, peaks[0], peaks[1], img_median)
        fout.write(os.path.basename(file) + '; ' + 
                   '; '.join(map(str, csv)).replace('.', ',') + '\n')
        fout.flush()
fout.close()