import glob
import os
import numpy as np
from skimage import io, transform
import math
from skimage.filters import gaussian, gabor
from random import randint, shuffle, uniform
import scipy
import cv2
import torch

auglevel = 5

def type_check(im):
    if isinstance(im, np.ndarray):
        return im
    elif isinstance(im, dict):
        return im['input']

def good_return(im, out):
    if isinstance(im, np.ndarray):
        return out
    elif isinstance(im, dict):
        im['input'] = out
        return im

def scale(arr):
    arr = arr - arr.min()
    arr = arr/(arr.max() + 0.000001)
    return arr

def clip(arr):
    arr = np.clip(arr, a_max=1.0, a_min=0)
    return arr

def brightness(inp):
    if auglevel == 0:
        return inp
    arr = type_check(inp)
    minb, maxb = (50-1*auglevel, 50+1*auglevel)
    p = randint(minb,maxb)
    arr = arr+scipy.special.logit(p/100)
    return good_return(inp, arr)

def contrast(inp):
    if auglevel == 0:
        return inp
    arr = type_check(inp)
    minc, maxc = 8-1*auglevel, 8+1*auglevel
    p = randint(minc, maxc)
    arr = arr*(p/10)
    return good_return(inp, arr)

def gamma(inp):
    if auglevel == 0:
        return inp
    arr = type_check(inp)
    ming, maxg = 10-1*auglevel, 10+1*auglevel
    gamma = randint(ming, maxg)/10
    arr = np.array(arr*255, dtype=np.uint8)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    arr = cv2.LUT(arr, table)/255
    return good_return(inp, arr)

def tanht(inp):
    if auglevel == 0:
        return inp
    arr = type_check(inp)
    arr = scale(arr)
    arr = (2 * arr - 1)

    a1 = np.random.uniform(-0.5, 0)
    a2 = np.random.uniform(0.2, 0.5)
    # a1 -> -.0.25
    # a2 -> 0.25
    w_1, b_1 = np.linalg.solve(
        a=[[a1, 1],
           [a2, 1]],
        b=[-0.25, 0.25]
    )

    b1 = np.random.uniform(-0.5, 0)
    b2 = np.random.uniform(0, 0.5)
    w_2, b_2 = np.linalg.solve(
        a=[[b1, 1],
           [b2, 1]],
        b=[-0.25, 0.25]
    )

    arr_orig = (np.arctanh(arr) - b_1) / (w_1 + 0.000000000001)
    arr = np.tanh(w_2 * arr_orig + b_2)

    # arr [-1, 1] -> [0, 1]
    arr = (arr + 1) / 2
    return good_return(inp, scale(arr))


def squeeze(inp, th1=0, th2=0):
    im = type_check(inp)
    if th1 ==0 and th2 ==0:
        th1 = 0.1*randint(0,4)
        th2 = 0.1*randint(5,10)
    gtarr = (im >= th2)*im
    ltarr = (im < th1)*im
    nim = ((im > th1)*(im < th2)*im)
    new_im = scale(1.2*ltarr+0.9*gtarr+nim)
    return good_return(inp, scale(new_im))

def sparse(inp, th1=0, th2=0):
    im = type_check(inp)
    if th1 ==0 and th2 ==0:
        th1 = 0.1*randint(0,4)
        th2 = 0.1*randint(5,10)
    gtarr = (im >= th2)*im
    ltarr = (im < th1)*im
    nim = ((im > th1)*(im < th2)*im)
    nim = range_scale(nim, 0.9*th1, 1.2*th2)
    new_im = scale(0.9*ltarr+1.2*gtarr+nim)
    return good_return(inp, scale(new_im))

def range_scale(im, th1=0, th2=0):
    non_zero_min = ((im == 0)*1+im).min()
    im_max = im.max()
    dif = th2 - th1
    im2 = im - non_zero_min
    im2 = (im2 > 0)*im2
    im2 = (im2 * (dif/(im2.max()+0.00001)))
    im2 = im2 + th1
    im2 = (im2 > th1)*im2
    return im2

def get_borders(im, th):
    x = im.shape[0]
    y = im.shape[1]
    mn = im.mean()
    bl,br,bu,bd = (0,x,0,y)
    for i in range(0, x):
        strip = im[i:i+1, :]
        if strip.std() > th or strip.mean() > mn/2:
            bl = i
            break
    for i in range(x, 0, -1):
        strip = im[i-1:i, :]
        if strip.std() > th or strip.mean() > mn/2:
            br = i
            break
    for i in range(0, y):
        strip = np.transpose(im[:, i:i+1], (1,0))
        if strip.std() > th or strip.mean() > mn/2:
            bu = i
            break
    for i in range(y, 0, -1):
        strip = np.transpose(im[:, i-1:i], (1,0))
        if strip.std() > th or strip.mean() > mn/2:
            bd = i
            break
    return (bl,br,bu,bd)

def randomcrop(size=224,h=0,w=0):
    def rc(inp):
        im = type_check(inp)
        new_im = im[h: h + size, w: w + size]
        return good_return(inp, scale(new_im))
    return rc

def centercrop(size=224):
    def cc(inp):
        im = type_check(inp)
        h,w = im.shape[0],im.shape[1]
        new_im = im[int((h-size)/2):int((h+size)/2),int((w-size)/2):int((w+size)/2)]
        return good_return(inp, scale(new_im))
    return cc

def rmblack(inp):
    im = type_check(inp)
    mn = im.mean()
    imarr = th_lower(im, 10)
    bds = get_borders(imarr, 0.2)
    imarr = im[bds[0]:bds[1], bds[2]:bds[3]]
    return good_return(inp, scale(imarr))

def th_lower(inp, p=-1):
    arr = type_check(inp)
    if p == -1:
        p = randint(0, 20)
    ll = np.percentile(arr, p)
    narr = arr - ll
    narr = (narr > 0)*narr
    narr = scale(narr)
    ul = np.percentile(narr, 99)
    img = (narr > ul)*ul
    iml = (narr < ul)*narr
    farr = scale(img + iml)
    return good_return(inp, farr)

def add_noise(inp, p=-1):
    arr = type_check(inp)
    if p == -1:
        p = 0.01*randint(2, 8)
    noise = np.random.normal(0, p, (arr.shape))
    new_arr = arr + noise
    return good_return(inp, scale(new_arr))

def smooth_noise(inp, p=-1):
    arr = type_check(inp)
    if p == -1:
        p = 0.01*randint(2, 8)
    noise = np.random.normal(0, p, (arr.shape))
    noise = gaussian(noise, 0.6)
    new_arr = arr + noise
    return good_return(inp, scale(new_arr))

def smooth(inp, p=-1):
    arr = type_check(inp)
    if p == -1:
        p = randint(5, 12)*0.1
    new_arr = gaussian(arr, p)
    return good_return(inp, scale(new_arr))

def mean_clip(inp):
    arr = type_check(inp)
    mn = arr.mean()
    new_arr = (arr > mn)*1
    return good_return(inp, scale(new_arr))

def lnorm(inp):
    arr = type_check(inp)
    arr = arr-arr.mean()
    arr = arr/(arr.std()+0.000001)
    return good_return(inp, arr)

def gaborr(inp, fq=-1):
    arr = type_check(inp)
    if fq == -1:
        fq = 0.1*randint(4,7)
    narr, narr_i = gabor(arr, frequency=fq)
    return good_return(inp, scale(narr))

def resize_hard(size=224):
    def rz(inp):
        im = type_check(inp)
        im = scale(im)
        new_im = cv2.resize(im, (size, size))
        return good_return(inp, scale(new_im))
    return rz

def hflip(p=0.5):
    px = int(p*100)
    def flip(arr):
        rint = randint(0,100)
        if rint < px:
            arr = np.fliplr(arr)
        return arr
    return flip

def resize_hard_sz(sz0, sz1):
    def rz(inp):
        if isinstance(inp, np.ndarray):
            im = inp
            im = scale(im)
            new_im = cv2.resize(im, (sz1, sz0))
            return new_im
        elif isinstance(inp, dict):
            im = {x:scale(inp[x]) for x in inp}
            new_im = {x:cv2.resize(im[x], (sz1, sz0)) for x in im}
            return new_im
    return rz

def sub_smooth(inp):
    im = type_check(inp)
    im = scale(im)
    nim = gaussian(im, 8)
    nim = im - nim
    nim = scale(nim)
    return good_return(inp, nim)

def scale_unity(inp):
    im = type_check(inp)
    im = scale(im)
    im = (im - 0.5)/0.5
    return good_return(inp, im)

def smooth_norm(inp):
    im = type_check(inp)
    blur1 = gaussian(im, 7)
    blur2 = gaussian(blur1, 7)
    blur3 = gaussian(blur2, 7)
    im = im - blur1 + blur2 - blur3
    return good_return(inp, scale(im))

def seg_randomsizedcrop(output_shape, frac_range=(0.5, 1)):
    def rzcrop(inp_dict):
        ims = [inp_dict[x] for x in inp_dict]
        shp = ims[0].shape
        rat = uniform(frac_range[0], frac_range[1])
        crop_dim = int(np.sqrt(ims[0].size*rat))
        stx, sty = randint(0, shp[0]-crop_dim-1), randint(0, shp[1]-crop_dim-1)
        inp_dict = {k:resize_hard(output_shape)(inp_dict[k][stx:stx+crop_dim, sty:sty+crop_dim]) for k in inp_dict}
        return inp_dict
    return rzcrop

def seg_randomcrop(output_shape):
    def rc(inp_dict):
        h, w = output_shape
        ims = [inp_dict[x] for x in inp_dict]
        im_h, im_w = ims[0].shape
        st_h, st_w = randint(0, im_h-h), randint(0, im_w-w)
        inp_dict = {k:inp_dict[k][st_h:st_h+h, st_w:st_w+w] for k in inp_dict}
        return inp_dict
    return rc

def tofloatten(arr):
    if arr.shape[-1] == 3:
        arr = np.rollaxis(arr, 2, 0)
    ten = torch.FloatTensor(arr)
    return ten