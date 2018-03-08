# -*- coding: utf-8 -*-
import cv2
import numpy as np

# from cax_models.python.color_category import skin_detector
import skin_detector


def resizing(img):
    height, width, channels = img.shape
    if max(height, width) > 100:
        ratio = float(height) / width
        new_width = 100 / ratio
        img_resized = cv2.resize(img, (int(new_width), 100))
        ip_convert = cv2.imencode('.png', img_resized)
    else:
        ip_convert = cv2.imencode('.png', img)

    return ip_convert


def edgedetect(channel):
    sobelx = cv2.Sobel(channel, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(channel, cv2.CV_16S, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    sobel[sobel > 255] = 255

    return sobel


def findSignificantContours(img, sobel_8u, sobel):
    image, contours, heirarchy = cv2.findContours(sobel_8u, \
                                                  cv2.RETR_EXTERNAL, \
                                                  cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(image.shape[:2], dtype="uint8") * 255

    level1 = []
    for i, tupl in enumerate(heirarchy[0]):

        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)
    significant = []
    tooSmall = sobel_8u.size * 10 / 100
    for tupl in level1:
        contour = contours[tupl[0]];
        area = cv2.contourArea(contour)
        if area > tooSmall:
            cv2.drawContours(mask, \
                             [contour], 0, (0, 255, 0), \
                             2, cv2.LINE_AA, maxLevel=1)
            significant.append([contour, area])
    significant.sort(key=lambda x: x[1])
    significant = [x[0] for x in significant];
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    mask = sobel.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255, 0)
    mask = np.logical_not(mask)
    img[mask] = 0;

    return img


def image_segmentation(ip_convert):
    img = cv2.imdecode(np.squeeze(np.asarray(ip_convert[1])), 1)

    
    # cv2.imwrite("Skin_removed.jpg",img_skin)

    height, width, channels = img.shape
#     blurred = cv2.GaussianBlur(img, (5, 5), 0)
#     
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (5, 5, width - 5, height - 5)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_mask = img* mask2[:, :, np.newaxis]
    cv2.imwrite("download(8)_grab.jpg",img_mask)
    # cv2.waitKey(0)
    # blurred = cv2.GaussianBlur(img_mask,(3,3),0)
    # img_skin = skin_detector.process(img_mask)
    # cv2.imwrite("download(10)_skin.jpg",img_skin)
    # cv2.waitKey(0)
    blurred = cv2.GaussianBlur(img_mask,(5,5),0)
    edgeImg = np.max( np.array([ edgedetect(blurred[:,:, 0]), edgedetect(blurred[:,:, 1]), edgedetect(blurred[:,:, 2]) ]), axis=0 )
    mean = np.mean(edgeImg);
# # Zero any value that is less than mean. This reduces a lot of noise.
    edgeImg[edgeImg < mean] = 0;
    edgeImg_8u = np.asarray(edgeImg, np.uint8)

# # Find contours
    significant = findSignificantContours(img_mask, edgeImg_8u, edgeImg)
    cv2.imwrite("download(8)_contour.jpg",significant)
    significant = cv2.GaussianBlur(significant,(3,3),0)
    tmp = cv2.cvtColor(significant, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 1, cv2.THRESH_BINARY)
    b, g, r = cv2.split(significant)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    img_out = cv2.imencode('.png', dst)
    # cv2.imshow("Masking_Done.jpg",dst)
    # cv2.waitKey(0)
    return img_out


def removebg(segmented_img):
    src = cv2.imdecode(np.squeeze(np.asarray(segmented_img[1])), 1)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    processed_img = cv2.imencode('.png', dst)

    return processed_img
