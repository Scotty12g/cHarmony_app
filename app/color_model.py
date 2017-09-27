def FindColor(matchtype, tomatch_path, closet_path):
  # Import the packages we'll need
    from sklearn.neighbors import KNeighborsClassifier
    from flask import Flask, request, render_template, url_for
    import pandas as pd
    import os
    from os.path import join
    import numpy as np
    import time
    import colorsys
    from PIL import Image, ImageEnhance
    import matplotlib.pyplot as plt
    import re
    from operator import itemgetter
    import random
    import cv2
    import skimage
    from skimage.transform import pyramid_gaussian
    from skimage.color import rgb2gray
    from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
    from skimage.segmentation import mark_boundaries, join_segmentations
    from skimage.exposure import adjust_gamma, equalize_hist
    from skimage.util import img_as_float
    from skimage import filters
    import math
    import sys

    from test_pick import Pickler
    # unpickling the model

    color_detector_unpickle = open('color_detector.pkl', 'r')
    color_detector = Pickler.load_pickle(color_detector_unpickle)
    color_detector_unpickle.close()
    
    # Path of the clothing you'd like to match with
    IMAGE_TO_MATCH_PATH = tomatch_path

    # Path of the mix of clothing to match to
    IMAGE_POSSIBLE_MATCHES_PATH = closet_path
    # How to match the color: 'complement' (180deg), 'triad' (120deg), or 'analogus' (30deg)
    HOW_TO_MATCH = matchtype
    
    # Now create a function to fix the color profiles
    def fix_color(img,percent_correct):
        # First we equalize the histogram, prividing a bit of contrast and correcting the spread of the distributions

        hist,bins = np.histogram(img.flatten(),256,[0,256])

        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        img2 = cdf[img]
    
        # Then we implement the color balance
        def apply_mask(matrix, mask, fill_value):
            masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
            return masked.filled()

        def apply_threshold(matrix, low_value, high_value):
            low_mask = matrix < low_value
            matrix = apply_mask(matrix, low_mask, low_value)

            high_mask = matrix > high_value
            matrix = apply_mask(matrix, high_mask, high_value)

            return matrix

        def simplest_cb(img, percent):
            assert img.shape[2] == 3
            assert percent > 0 and percent < 100

            half_percent = percent / 200.0

            channels = cv2.split(img)

            out_channels = []
            for channel in channels:
                assert len(channel.shape) == 2
                # find the low and high precentile values (based on the input percentile)
                height, width = channel.shape
                vec_size = width * height
                flat = channel.reshape(vec_size)

                assert len(flat.shape) == 1

                flat = np.sort(flat)

                n_cols = flat.shape[0]

                low_val  = flat[int(math.floor(n_cols * half_percent))]
                high_val = flat[int(math.ceil( n_cols * (1.0 - half_percent)))]

                # saturate below the low percentile and above the high percentile
                thresholded = apply_threshold(channel, low_val, high_val)
                # scale the channel
                normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
                out_channels.append(normalized)

            return cv2.merge(out_channels)
        return simplest_cb(img2,percent_correct)


    
    # Import and IMAGE_TO_MATCH and convert to RBG from BGR
    IMAGE_TO_MATCH = cv2.imread(IMAGE_TO_MATCH_PATH)
    IMAGE_TO_MATCH = cv2.cvtColor(IMAGE_TO_MATCH, cv2.COLOR_BGR2RGB)
    basewidth = 600
    wpercent = (basewidth/float(IMAGE_TO_MATCH.shape[0]))
    hsize = int((float(IMAGE_TO_MATCH.shape[1])*float(wpercent)))
    IMAGE_TO_MATCH = cv2.resize(IMAGE_TO_MATCH,dsize=(hsize,basewidth), interpolation = cv2.INTER_AREA)
    IMAGE_TO_MATCH=fix_color(IMAGE_TO_MATCH,10)

    # Import and IMAGE_POSSIBLE_MATCHES_PATH and convert to RBG from BGR
    IMAGE_POSSIBLE_MATCHES = cv2.imread(IMAGE_POSSIBLE_MATCHES_PATH)
    IMAGE_POSSIBLE_MATCHES = cv2.cvtColor(IMAGE_POSSIBLE_MATCHES, cv2.COLOR_BGR2RGB)
    basewidth = 600
    wpercent = (basewidth/float(IMAGE_POSSIBLE_MATCHES.shape[0]))
    hsize = int((float(IMAGE_POSSIBLE_MATCHES.shape[1])*float(wpercent)))
    IMAGE_POSSIBLE_MATCHES = cv2.resize(IMAGE_POSSIBLE_MATCHES,dsize=(hsize,basewidth), interpolation = cv2.INTER_AREA)
    IMAGE_POSSIBLE_MATCHES=fix_color(IMAGE_POSSIBLE_MATCHES,10)
    
    # setup parameters to explore the space
    maxX = IMAGE_TO_MATCH.shape[1]
    maxY = IMAGE_TO_MATCH.shape[0]
    one_ycord = maxY/100.0
    one_xcord = maxY/100.0

    # Select 400 random points from the center of the image
    ycord = (np.random.choice(np.arange(one_ycord*40,one_ycord*60,1),400)).astype(int)
    xcord = (np.random.choice(np.arange(one_xcord*40,one_xcord*60,1),400)).astype(int)
    
    
    # Itterate through each point and predict the color
    colvec = []
    for i in np.arange(len(ycord)):
        newcol = color_detector.predict(np.array(IMAGE_TO_MATCH[ycord[i],xcord[i],:]).reshape(1, -1))
        colvec.append(newcol)

    colvec = np.vstack( colvec ).flatten()

    unique,pos = np.unique(colvec,return_inverse=True)
    counts = np.bincount(pos)
    maxpos = counts.argmax()
    IMAGE_TO_MATCH_OUTPUT = unique[maxpos]
    
    # Determine the opposite color
    color_dict = {'red':(255, 0, 0),
                'yellow' : (255, 255, 0),
                'green' : (0, 255, 0),
                'cyan' : (0, 255, 255),
                'blue' : (0, 0, 255),
                'magenta' : (255, 0, 255)}



    # a function to rotate the HLS color wheel, based on what type of matching you want
    HOW_TO_MATCH_dict = {'complement': 180,
                         'triad': 120}

    def rotate_colors((r, g, b), degs): # Assumption: r, g, b in [0, 255]
        d = degs/360.0
        r, g, b = map(lambda x: x/255., [r, g, b]) # Convert to [0, 1]
        h, l, s = colorsys.rgb_to_hls(r, g, b)     # RGB -> HLS
        h = [(h+d) % 1 for d in (-d, d)]           # Rotation by d
        rotated = [map(lambda x: int(round(x*255)), colorsys.hls_to_rgb(hi, l, s))
                for hi in h] # H'LS -> new RGB
        return rotated

    # Create a list of the RGB profiles for colors that match, then get the color names for them
    # If it's a neutral color, return all colors except the one you're matching
    # NOTE: only supports ONE color from the IMAGE_TO_MATCH at the moment
    if IMAGE_TO_MATCH_OUTPUT in ['black', 'brown', 'grey', 'white']:
        COLOR_MATCH = np.unique(all_Labels).tolist()
        COLOR_MATCH = list(filter(lambda x: x!= IMAGE_TO_MATCH_OUTPUT, COLOR_MATCH))
    else:
        MATCHING_COLORS=rotate_colors(color_dict[IMAGE_TO_MATCH_OUTPUT],HOW_TO_MATCH_dict[HOW_TO_MATCH])
        MATCHING_COLOR_LIST = list()
        for sublist in MATCHING_COLORS:
            if sublist not in MATCHING_COLOR_LIST:
                MATCHING_COLOR_LIST.append(sublist)

        COLOR_MATCH = [col for col in color_dict if color_dict[col] in tuple(tuple(x) for x in MATCHING_COLOR_LIST)]
 
    
    ### Input Image of Possible Matches (multiple colors) ###
    IMAGE_TO_SEGMENT=adjust_gamma(IMAGE_POSSIBLE_MATCHES, gamma=.75, gain=1)
    # try a bilateral filter, which is effective at noise removal 
    # while preserving edges, hopefully breaking up some confusing pattern
    IMAGE_TO_SEGMENT = cv2.bilateralFilter(IMAGE_TO_SEGMENT,5,75,75)

    ## OK, i'm choosing SLIC segmentation for now. Now I need to figure out how to sumamrize by image segemnt
    segments_slic = slic(IMAGE_TO_SEGMENT, n_segments=350, compactness=10, sigma=1)
    
    # Get the unique segments, and then summarise (mode) by segment for RBG
    unique_segments = np.unique(segments_slic)
    segment_summary = []
    for segment in unique_segments:
        image_segment = IMAGE_TO_SEGMENT[segments_slic==segment]
        image_segment_sub = [image_segment[i] for i in np.random.choice(np.arange(len(image_segment)),(len(image_segment)/5),replace=False)]
        segment_summary.append(image_segment_sub)

    # Predict the color of each segment
    SEGMENT_COLORS=[]
    for x in segment_summary:
        if len(np.array(x).shape)==1:
            if np.array(x).size == 0:
                x
            else:
                colors_seen=color_detector.predict(np.array(x).reshape(1, -1))
        else:
            colors_seen=color_detector.predict(x)
        unique,pos = np.unique(colors_seen,return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        SEGMENT_COLORS.append((unique[maxpos],counts[maxpos])[0])

    # create a boolean for which segemnts are the correct color, and subset the segments that match
    SEGMENT_HIGHLIGHT_BOOL = np.reshape([(val in COLOR_MATCH) for val in SEGMENT_COLORS],np.array(SEGMENT_COLORS).shape)
    SEGMENT_HIGHLIGHT = unique_segments[SEGMENT_HIGHLIGHT_BOOL]

    # Now create another image with all the "matching" colors highlighted in yellow
    HIGHLIGHT_MASK = np.zeros(IMAGE_TO_SEGMENT.shape, dtype=np.uint8)


    for y in np.arange(segments_slic.shape[0]): # from 0 to height at a step size
        for x in np.arange(segments_slic.shape[1]):
            if segments_slic[y,x] in SEGMENT_HIGHLIGHT:
                HIGHLIGHT_MASK[y,x,0] = 255
                HIGHLIGHT_MASK[y,x,1] = 255
                HIGHLIGHT_MASK[y,x,2] = 255
    
    IMAGE_TO_SEGMENT2=Image.fromarray(np.uint8(IMAGE_TO_SEGMENT))
    HIGHLIGHT_MASK2=Image.fromarray(np.uint8(HIGHLIGHT_MASK))
    img1 = IMAGE_TO_SEGMENT2.convert("RGBA")
    img2 = HIGHLIGHT_MASK2.convert("RGBA")
    datas = img2.getdata()

    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append((0, 0, 0, 255))

    img2.putdata(newData)

    img1.paste(img2, mask=img2)
    img1=img1.convert("RGBA")
    IMAGE_BLEND=Image.blend(IMAGE_TO_SEGMENT2.convert("RGBA"), img1, .85)
    IMAGE_BLEND=IMAGE_BLEND.convert("RGB")
    #IMAGE_BLEND.save(os.path.join(app.config['DOWNLOAD_FOLDER'],'result.png'))
    return COLOR_MATCH, IMAGE_TO_MATCH_OUTPUT,IMAGE_BLEND
