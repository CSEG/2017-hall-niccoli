import numpy as np
import scipy as sp
from scipy import ndimage as ndi
from skimage import color, exposure, feature
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, disk, square, opening, dilation
from skimage.measure import find_contours, approximate_polygon
from skimage import segmentation
from skimage.future import graph
from skimage import transform as tf
from skimage.draw import polygon


def find_largest(img, threshold = 'opening'):
    """Function to find largest bright object in an image. 
    Workflow:
    -   Convert to binary
    -   Find and retain only largest object in binary image
    -   Fill holes
    -   Apply opening and dilation to remove minutiae
    
    Current options to convert to binary:
    -   'opening':   applies global Otsu thresholding after a minor morphological opening (default)
    -   'global':    applies global Otsu thresholding
    -   'adaptive':  applies adaptive threshold after gaussian smoothing
    -   'tips':      eliminates both ~black and ~nearly white backgrounds
    -   'rag':       applies RAG thresholding, followed by global Otsu. Slowest method.
    
    Coming soon:
    -   'textural':  eliminates backgrounds of (any) constant colour based on GLCM or binary patterns
    -   'bilateral': applies adaptive threshold, but after bilateral smoothing"""
    

    # Gray scale conversion and contrast stretching
    gry = color.rgb2gray(img);
    p2, p95 = np.percentile(gry, (2, 95))
    
    if threshold == 'rag':
        labels1 = segmentation.slic(img, compactness=50, n_segments=500) 
        out1 = color.label2rgb(labels1, img, kind='avg')
        thresh =  min(np.shape(out1)[:1])/10 # threshold = 1/10th of the smallest side of the image
                                             # with a large value we get few colours
        g = graph.rag_mean_color(img, labels1)
        labels2 = graph.cut_threshold(labels1, g, thresh)
        out2 = color.label2rgb(labels2, img, kind='avg')
        out2 = color.rgb2gray(out2);
        global_thresh = threshold_otsu(out2)
        binary = out2 < global_thresh
   
    elif threshold == 'tips':     
        rescale = exposure.rescale_intensity(gry, in_range=(p2, p95))
        binary = np.logical_and(color.rgb2gray(rescale) > 0.05, color.rgb2gray(rescale) < 0.95)     
        
    elif threshold == 'adaptive': 
        gry = sp.ndimage.filters.gaussian_filter(color.rgb2gray(gry),7) 
        rescale = exposure.rescale_intensity(gry, in_range=(p2, p95))
        thresh_sauvola = threshold_sauvola(rescale, window_size=99)
        binary = rescale < thresh_sauvola
    
    elif threshold == 'global': 
        rescale = exposure.rescale_intensity(gry, in_range=(p2, p95))
        global_thresh = threshold_otsu(rescale)
        binary = rescale < global_thresh
    
    else:
        rescale = 1-opening(1-(exposure.rescale_intensity(gry, in_range=(p2, p95))), disk(2))
        global_thresh = threshold_otsu(rescale)
        binary = rescale < global_thresh 


    
    # Detect largest bright element in the binary image. Making the assumption it would be the map.
    # Eliminate everything else (text, colorbar, holes, ...).
    # Label all white objects (made up of ones)
    label_objects, nb_labels = ndi.label(binary) # ndimage.label actually labels 0 (background) as 0 and then 
                                                        # labels every nonzero object as 1, 2, ... n. 
    # Calculate every labeled object's size. 
    # np.bincount ignores whether input is an image or another type of array.
    # It just calculates the binary sizes, including for the 0 (background).
    sizes = np.bincount(label_objects.ravel())   
    sizes[0] = 0    # This sets the size of the background to 0 so that if it happened to be larger than 
                    # the largest white object it would not matter
   
    # Keep only largest object
    binary_objects = remove_small_objects(binary, max(sizes)) 
    
    # Remove holes from it (black regions inside white object)
    binary_holes = ndi.morphology.binary_fill_holes(binary_objects) 
    
    # This may help remove minutiae on the outside (tick marks and tick labels)
    binary_mask = opening(binary_holes, disk(7))
    return binary_mask



def ordered(points):
    """Function to sort corners based on angle from centroid. 
       Modified from: http://stackoverflow.com/a/31235064/1034648"""
    x = points[:,0]
    y = points[:,1]
    cx = np.mean(x)
    cy = np.mean(y)
    a = np.arctan2(y - cy, x - cx)
    order = a.ravel().argsort()
    x = x[order]
    y = y[order]
    return np.vstack([x,y])


def rectify_seismic(img, binary_mask):
    """Function to warp to a rectangle the area in the input img defined by binary_mask 
    It returns the warped area as an image"""
    
    # Find mask contour, approximate it with a quadrilateral, find and sort corners
    contour = np.squeeze(find_contours(binary_mask, 0))
    coords = approximate_polygon(contour, tolerance=50)
    
    # sort the corners with the exception of the last one (repetition of first corner)
    sortedCoords = ordered(coords[:-1]).T
    
    # Define size of output image based on largest width and height in the input
    w1 = np.sqrt(((sortedCoords[0, 1]-sortedCoords[3, 1])**2)+((sortedCoords[0, 0]-sortedCoords[3, 0])**2))
    w2 = np.sqrt(((sortedCoords[1, 1]-sortedCoords[2, 1])**2)+((sortedCoords[1, 0]-sortedCoords[2, 0])**2))
    h1 = np.sqrt(((sortedCoords[0, 1]-sortedCoords[1, 1])**2)+((sortedCoords[0, 0]-sortedCoords[1, 0])**2))
    h2 = np.sqrt(((sortedCoords[3, 1]-sortedCoords[2, 1])**2)+((sortedCoords[3, 0]-sortedCoords[2, 0])**2))
    w = max(int(w1), int(w2))
    h = max(int(h1), int(h2))
    
    # Define rectangular destination coordinates (homologous points) for warping
    dst = np.array([[0, 0],
                    [h-1, 0],
                    [h-1, w-1],
                    [0, w-1]], dtype = 'float32')
    
    # Estimate warping transform, apply to input image (mask portion), and output
    dst[:,[0,1]] = dst[:,[1,0]]
    sortedCoords[:,[0,1]] = sortedCoords[:,[1,0]]
    tform = tf.ProjectiveTransform()
    tform.estimate(dst,sortedCoords)
    warped = tf.warp(img, tform, output_shape=(h-1, w-1))
    return warped


def auto_canny(img, sigma = 0.33):
    """Zero-parameter, automatic Canny edge detection using scikit-image.
    Original function from pyimagesearch: Zero-parameter, automatic Canny edge with with Python and OpenCV
    www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv"""
        
    # compute the median of the single channel pixel intensities
    v = np.median(img)
 
    # apply automatic Canny edge detection using the computed median
    lower = float(max(0.0, (1.0 - sigma) * v))
    upper = float(min(1.0, (1.0 + sigma) * v))
    edged = feature.canny(img, sigma, lower, upper)

    # return the edged image
    return edged

# Function to test if 4 points form a rectangle. Adapted to Python from: 
# http://stackoverflow.com/questions/2303278/find-if-4-points-on-a-plane-form-a-rectangle/2304031#2304031
def is_rectangle(points):
    x = points[:,0]
    y = points[:,1]
    cx = np.mean(x)
    cy = np.mean(y)
    dd1 = abs(cx-x[0]) + abs(cy-y[0])
    dd2 = abs(cx-x[1]) + abs(cy-y[1])
    dd3 = abs(cx-x[2]) + abs(cy-y[2])
    dd4 = abs(cx-x[3]) + abs(cy-y[3])
    return abs(dd1-dd2)<=10 and abs(dd1-dd3)<=10 and abs(dd1-dd4)<=10
    # A better alternative may be to use angles between the semi-diagonals connecting centroid with 4 corners (sorted).
    
    
def remove_annotations(img, g = None):
    """Function to remove rectangular areas with annotations from an image.
    Workflow:
    -   Blur a bit and find canny edges
    -   Find contours of edges and approximate
    -   Retain only approximated contours (rectangles) based on two conditions: 
             having 4 sides and having 4 equal semi-diagonals
    -   Convert retained contours to polygons
    - Add to a binary mask
    - Use mask to remove the rectangular areas in the input"""
    
    if g is not None:
        gry = sp.signal.medfilt2d(color.rgb2gray(img), g)
    else: 
        gry = color.rgb2gray(img)
    #
    cn = auto_canny(color.rgb2gray(gry)) # canny edges with automatic parameters based on median pixel intensity
    cny_dilated = dilation(cn, disk(2.5)) # dilate edges
    
    mask = np.zeros(np.shape(cny_dilated), dtype=np.uint8)
    for contour in find_contours(cny_dilated, 0):
        coords = approximate_polygon(contour, tolerance=30)
        if len(coords) == 5 and is_rectangle(ordered(coords[:-1]).T):
            rr,cc = polygon(ordered(coords[:-1])[0], ordered(coords[:-1])[1])
            mask[rr,cc]=1
    
    dmask = dilation(mask, square(10))
    
    image = img.copy()
    for layer in range(image.shape[-1]):
        image[np.where(dmask)] = np.NaN
    return image
   