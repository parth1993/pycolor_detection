# -*- coding: utf-8 -*-
import operator
from collections import defaultdict
from math import sqrt
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import matplotlib.colors as colors
import numpy as np
import numpy.ma as ma
from colormap import rgb2hex
from sklearn.cluster import KMeans
from multiprocessing import Pool, cpu_count
import matplotlib.cm as cm

'__author__' == "sharmaparth17@gmail.com"

random.seed(0)
def best_Clust(img, clusters):
    # img, clusters = arg
    clt = KMeans(n_clusters=clusters, random_state=2, n_jobs=1)
    clt.fit(img)

    return ([clusters, clt.inertia_])


def func(t):
    if (t > 0.008856):
        return np.power(t, 1 / 3.0)
    else:
        return 7.787 * t + 16 / 116.0


def rgbtolab(requested_color):
    # Conversion Matrix
    matrix = [[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]]

    # RGB values lie between 0 to 1.0
    cie = np.dot(matrix, requested_color);
    cie[0] = cie[0] / 0.950456;
    cie[2] = cie[2] / 1.088754;
    # Calculate the L
    L = 116 * np.power(cie[1], 1 / 3.0) - 16.0 if cie[
                                                      1] > 0.008856 else 903.3 * \
                                                                         cie[
                                                                             1];

    # Calculate the a
    a = 500 * (func(cie[0]) - func(cie[1]));

    # Calculate the b
    b = 200 * (func(cie[1]) - func(cie[2]));

    #  Values lie between -128 < b <= 127, -128 < a <= 127, 0 <= L <= 100
    Lab = [L, a, b];

    # OpenCV Format
    L = L * 255 / 100;
    a = a + 128;
    b = b + 128;
    Lab_OpenCV = [b, a, L];

    return Lab


def match_colour(ccolor):
    # converting hex to rgb

    requested_color = colors.hex2color(ccolor.upper())

    lab = rgbtolab(requested_color)

    return lab


def data(dfs):
    dfs['Hex_code'] = dfs['Hex_code'].apply(str)
    dfs['hex_to_rgb'] = dfs['Hex_code'].apply(match_colour)

    return dfs


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
    # return the bar chart
    return bar


def closest_colour(requested_colour, dfs):
    min_colours = {}

    requested_color = colors.hex2color(requested_colour)

    requested_color = rgbtolab(requested_color)
    for key, color_shade, code, color_base in zip(dfs['hex_to_rgb'] \
            , dfs['Color_shade'], dfs['Hex_code'], dfs['Color_category']):
        r_c, g_c, b_c = map(float, key)
        rd = abs(r_c - float(requested_color[0]))
        gd = abs(g_c - float(requested_color[1]))
        bd = abs(b_c - float(requested_color[2]))
        min_colours[sqrt(rd + gd + bd)] = [color_shade, code, color_base]
    return min_colours[min(min_colours.keys())]


def detect_color(img_byte_array, map_path):
    color_csv = []
    actualhexcod_csv = []
    closest_nameshade = dict()
    closest_namebase = defaultdict(list)
    cluster_errors = []
    dom_array = []

    dfs = data(map_path)
    final_colorcsv = defaultdict(list)
    img = cv2.imdecode(np.squeeze(np.asarray(img_byte_array[1])), -1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_reshaped = img_rgb.reshape((img_rgb.shape[0] * img_rgb.shape[1], 3))
    img_final = ma.masked_where(img_reshaped == [0, 0, 0], img_reshaped)

    pool = Pool(processes=12)
    results = [pool.apply_async(best_Clust, (img_final, w)) for w in
               range(3, 13)]

    cluster_errors = [p.get() for p in results]
    dic = dict()
    for i in cluster_errors:
        dic[i[1]] = i[0]
    bestCluster = dic[min(dic.keys())]
    pool.close()
    pool.join()
    clt = KMeans(n_clusters=bestCluster, random_state=2, n_jobs=-1)
    clt.fit(img_final)
    hist = centroid_histogram(clt)
    # centroids = clt.cluster_centers_
    # color = clt.labels_
    # fig = plt.figure()
    # ax = Axes3D(plt.gcf())
    # x = np.arange(100,200,1)
    # y = np.arange(100,200,1)
    # dat = plt.cm.jet(clt.labels_)

    # ax.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2], marker = "o", s=150, linewidths = 5, zorder = 100, c='red')

    # print (clt.cluster_centers_)
    # plt.show()
    closest_hex = dict()
    closest_shade_with_hex = dict()

    for (percent, color) in zip(hist, clt.cluster_centers_):

        requested_colour = color
        if int(requested_colour[0]) > 0 and int(
                requested_colour[1]) > 0 and int(
            requested_colour[2]) > 0:
            hexcod = rgb2hex(int(requested_colour[0]), \
                             int(requested_colour[1]), \
                             int(requested_colour[2]))
            actualhexcod_csv.append(hexcod)
            output = closest_colour(hexcod, dfs)
            color_shade, hexcoda, color_base = output[0], output[1], output[2]
            color_csv.append(requested_colour)
            closest_hex[hexcod] = percent * 100
            if color_base not in closest_nameshade.keys():
                closest_nameshade[color_base] = percent * 100
                closest_namebase[color_base] += [[color_shade, percent * 100]]
                closest_shade_with_hex[color_base] = {'shade':color_shade, 'hex': hexcod, 'percent': percent * 100}
            else:
                closest_nameshade[color_base] += percent * 100
                closest_shade_with_hex[color_base] = {'shade':None, 'hex': hexcod, 'percent': percent * 100}
    # sorted_closest_hex = sorted(closest_hex.items(), key=operator.itemgetter(1))
    # sorted_shade_with_hex = sorted(closest_shade_with_hex, key=lambda x: (closest_shade_with_hex[x]['percent']), reverse=True)
    final_shade = list()
    final_percent = list()
    
    for key in closest_nameshade.keys():
        final_colorcsv[closest_nameshade[key]] = key


    tmp = sorted(final_colorcsv.keys())
    tmp.reverse()


    if len(tmp) == 1:
        final_shade.append([final_colorcsv[tmp[0]], \
                            max(closest_namebase[final_colorcsv[tmp[0]]], \
                                key=lambda x: x[1])])

    else:

        for key in range(len(tmp)):
            if ('white' == final_colorcsv[tmp[key]] or 'gray' == final_colorcsv[tmp[key]]) and tmp[key] < 13:
                continue
            if tmp[key] > 3.5:
                final_shade.append([final_colorcsv[tmp[key]], \
                                    max(closest_namebase[final_colorcsv[tmp[key]]], \
                                        key=lambda x: x[1])])
    if final_shade == []:
        final_shade.append([final_colorcsv[tmp[0]], \
                            max(closest_namebase[final_colorcsv[tmp[0]]], \
                                key=lambda x: x[1])])
    for B, S in final_shade:
        if len(final_shade) > 1:
            dom_array.append(
                {
                    'prediction':
                        {
                            'Shade': str(S[0]).title(),
                            'Base': str(B).title(),
                        },
                    'hex': closest_shade_with_hex[str(B)].get('hex'),
                    'probability': 1
                }
            )
            continue
        if len(final_shade) == 1:
            dom_array.append(
                {
                    'prediction':
                        {
                            'Shade': str(S[0]).title(),
                            'Base': str(B).title(),
                        },
                    'hex': closest_shade_with_hex[str(B)].get('hex'),
                    'probability': 1
                }
            )
            continue
    if len(final_shade) == 0:
        final_shade.append([final_colorcsv[key], \
                            max(closest_namebase[final_colorcsv[key]])])
        dom_array.append(
            {
                'prediction':
                    {
                        'Shade': str(final_shade[1][0]).title(),
                        'Base': str(final_shade[0]).title(),
                    },
                    'hex': closest_shade_with_hex[str(final_shade[0])].get('hex'),
                'probability': 1
            }
        )
    return dom_array

