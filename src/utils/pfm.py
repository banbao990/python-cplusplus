# https://github.com/microsoft/AirSim/blob/main/PythonClient/airsim/utils.py

import numpy as np  # pip install numpy
import sys
import re
import logging


def read_pfm(file):
    """ Read a pfm file """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    header = str(bytes.decode(header, encoding='utf-8'))
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    temp_str = str(bytes.decode(file.readline(), encoding='utf-8'))
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', temp_str)
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    # DEY: I don't know why this was there.
    file.close()

    return data, scale


def write_pfm(file, image, scale=1, endian='<'):
    """ Write a pfm file """
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    # grayscale
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    temp_str = '%d %d\n' % (image.shape[1], image.shape[0])
    file.write(temp_str.encode('utf-8'))

    # little-endian
    if endian == '<':
        scale = -scale

    temp_str = '%f\n' % scale
    file.write(temp_str.encode('utf-8'))

    image.tofile(file)


def write_png(filename, image):
    """ image must be numpy array H X W X channels
    """
    import cv2      # pip install opencv-python

    ret = cv2.imwrite(filename, image)
    if not ret:
        logging.error(f"Writing PNG file {filename} failed")
