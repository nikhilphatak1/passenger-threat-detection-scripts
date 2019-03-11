import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math


def read_header(infile):
    """Read image header (first 512 bytes)
    """
    h = dict()
    fid = open(infile, 'r+b')
    h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
    h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
    h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
    h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_polarization_channels'] =np.fromfile(fid, dtype = np.int16,count = 1)
    h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
    h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
    h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
    h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
    h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
    h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['elevation_offset_angle'] = np.fromfile(fid,dtype = np.float32, count = 1)
    h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
    h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
    h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)
    return h

def read_data(infile):
    """
    Read any of the 4 types of image files,
    returns a numpy array of the image contents
    """
    extension = os.path.splitext(infile)[1]
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    fid = open(infile, 'rb')
    fid.seek(512) #skip header
    if extension == '.aps' or extension == '.a3daps':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, ny, nt, order='F').copy() #make N-d image
    elif extension == '.a3d':
        if(h['word_type']==7): #float32
            data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
        elif(h['word_type']==4): #uint16
            data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)
        data = data * h['data_scale_factor'] #scaling factor
        data = data.reshape(nx, nt, ny, order='F').copy() #make N-d image
    elif extension == '.ahi':
        data = np.fromfile(fid, dtype = np.float32, count = 2* nx * ny * nt)
        data = data.reshape(2, ny, nx, nt, order='F').copy()
        real = data[0,:,:,:].copy()
        imag = data[1,:,:,:].copy()
    fid.close()
    if extension != '.ahi':
        return data
    else:
        return real, imag

def bwmorph_endpoints( im ):

    retval, im = cv2.threshold( im, 128, 1, cv2.THRESH_BINARY )

    lut_endpoints = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,1,
                                1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
                                0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,
                                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,
                                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,
                                1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,
                                0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
                                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
                                0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,
                                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,
                                1,0,1,1,1,1,0], dtype=np.uint8 )

    kernel = 2 ** np.array( range(9) )
    kernel = kernel.reshape( (3,3) )

    dst = im.copy()
    dst = cv2.filter2D( dst, cv2.cv.CV_16UC1, kernel )

    dst = lut_endpoints[ dst ]

    return dst.astype(np.uint8)

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


'''
da90 = '/home/nikhilphatak1/Documents/kaggle/data/stage1%2Fa3d%2F00360f79fd6e02781457eda48f85da90.a3d'
x = read_data(da90)
G=x[:,:,200]
#print(G[200])
del x



sliceThrowoutThreshold = 3.5e-4
groupDistanceThreshold = 80
weighted = False

#currentSliceName =
           #'slice{0}.mat'.format(sliceNum - 1)
maxVal = np.amax(G)
if (not (maxVal > sliceThrowoutThreshold)):
    G[:,:] = 0

    #return (G,0,[])

    # Plot original
    #if (plot):
        # construct figure here I guess

    # canny edge detection


#print(maxVal)

G = G / maxVal
G = G*256
G = np.uint8(G)

#print(np.amax(G))
# .12,.3
G = cv2.GaussianBlur(G, (7,7), math.sqrt(2))
G = cv2.Canny(G,33,77)

    # # bwareaopen equivalent function on G_canny here


se = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))
G = cv2.morphologyEx(G,cv2.MORPH_CLOSE,se)

    # # bwmorph 'thin' transform

    # # bwmorph 'spur' and 'thin'


fig = plt.figure()
ax = fig.add_subplot(111)
plt.axis([0,G.shape[1],0,G.shape[0]])
plt.pcolor(G,cmap='viridis')
plt.colorbar()
plt.show()
'''

# returns suspicionMatrix, outputImage, numRegions, regionCentroids
def filterSlice(sliceNum):
    sliceThrowoutThreshold = 3.5e-4
    groupDistanceThreshold = 80
    minAllowableSpurLength = 6
    branchMetricSearchLength = 25;
    endPtSuspNormalizationFactor = 5;
    angSuspNormalizationFactor = 3000*3;
    suspicionSmearingSigma = 2

    da90 = '/home/nikhilphatak1/Documents/kaggle/data/stage1%2Fa3d%2F00360f79fd6e02781457eda48f85da90.a3d'
    x = read_data(da90)
    G=x[:,:,sliceNum]
    del x
    maxVal = np.amax(G)

    if maxVal > sliceThrowoutThreshold:
        G = G / maxVal
        G = G*256
        G = np.uint8(G)
    else:
        G[:,:] = 0

        outputImage = G
        numRegions = 0
        regionCentroids = []
        return


    G = cv2.GaussianBlur(G, (7,7), math.sqrt(2))
    G_canny = cv2.Canny(G,33,77)

    # G_canny = bwareaopen(G_canny,10) #

    se = cv2.getStructuringElement(cv2.MORPH_RECT,(6,6))
    G_canny = cv2.morphologyEx(G_canny,cv2.MORPH_CLOSE,se)

    se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    G_canny = cv2.morphologyEx(G_canny,cv2.MORPH_OPEN,se)


    # G_skel = bwmorph(G_canny,'thin',inf) #

    # G_skel = bwareaopen(G_skel,10) #

    #despur not yet implemented
    G_skel, endPointRows, endPointCols, endPointBearings, branchPointRows, branchPointCols, branchPointMetrics = despur(G_skel, minAllowableSpurLength, branchMetricSearchLength)
    G_skel = np.ones(512)
    outputImage = G_skel

    orientations,orientationMetrics = skeletonOrientation12_06(G_skel,[5,5],branchPointRows,branchPointCols);

    orientationMetrics = orientationMetrics/angSuspNormalizationFactor

    distThresh = 20

    # labeledImage = bwlabel(G_skel,8) #

    # need to define sub2ind endPointGroups = labeledImage[sub2ind(labeledImage.shape,endPointRows,endPointCols)] #

    numEndPoints = len(endPointRows)

    metric = np.zeros(numEndPoints,1)

    if numEndPoints > 1:
        for i in range(numEndPoints):
            currentGroup = labeledImage[endPointRows[i],endPointCols[i]]
            distances = math.sqrt((endPointRows - endPointRows[i])**2 + (endPointCols - endPointCols[i])**2)
            distancesNew = distances.sort(axis=1)
            indices = distances.argsort(axis=1)
            distances = distancesNew

            distances[0] = []
            indices[0] = []

            #distances(endPointGroups(indices) == currentGroup) = [] #
            #indices(endPointGroups(indices) == currentGroup) = [] #

            if isempty(distances):
                continue

            if distances[0] < distThresh:
                pt1row = endPointRows[i]
                pt1col = endPointCols[i]
                pt1theta = endPointBearings[i]

                pt2row = endPointRows[indices[0]]
                pt2col = endPointCols[indices[0]]
                pt2theta = endPointBearings[indices[0]]

                if pt2theta > 0:
                    pt2theta = pt2theta - 180
                else:
                    pt2theta = pt2theta + 180

                segmentTheta = atan2d(pt2row-pt1row,pt2col-pt1col)

                angleChange1 = segmentTheta - pt1theta

                if angleChange1 > 180:
                    angleChange1 = angleChange1 - 360
                elif angleChange1 < -180:
                    angleChange1 = angleChange1 + 360

                angleChange2 = pt2theta - segmentTheta

                if angleChange2 > 180:
                    angleChange2 = angleChange2 - 360
                elif angleChange2 < -180:
                    angleChange2 = angleChange2 + 360

                metric[i] = math.abs(angleChange2-angleChange1) / max(math.abs(angleChange2+angleChange1),0.1)



    metric = metric / endPtSuspNormalizationFactor

    suspicionMatrix = np.zeros(outputImage.shape)
    if endPointRows.size == 0:
        suspicionMatrix[sub2ind(suspicionMatrix.shape,endPointRows,endPointCols)] = metric

    if branchPointRows.size == 0:
        suspicionMatrix[sub2ind(suspicionMatrix.shape,branchPointRows,branchPointCols)] = branchPointMetrics

    suspicionMatrix[outputImage == 1] = suspicionMatrix[outputImage==1] + orientationMetrics[outputImage==1]


    suspicionMatrix = cv2.GaussianBlur(suspicionMatrix,(3,3),suspicionSmearingSigma)

    suspicionMatrix[outputImage==1] = -1 / cv2.countNonZero(outputImage)
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis([0,G_canny.shape[1],0,G_canny.shape[0]])
    plt.pcolor(G_canny,cmap='gray')
    plt.colorbar()
    plt.show()

filterSlice(200)
