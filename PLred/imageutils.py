import numpy as np
import matplotlib.pyplot as plt
# import cv2
from scipy.ndimage import label, center_of_mass
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt

def shift_image_warpaffine(im, shift_x, shift_y):
    ''' shift image using warpAffine '''
    import cv2
    # shift the image
    M = np.float32([[1,0,shift_x], [0,1,shift_y]])
    shifted = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))

    return shifted

def shift_image_fourier(im0, shift_x, shift_y, oversample_factor = 2, npad = 10):
    ''' shift image using Fourier transform '''

    from scipy.fft import fft2, ifft2, fftshift
    from scipy.ndimage import zoom

    im = zoom(im0, oversample_factor)
    padded = np.pad(im, npad)

    nx, ny = padded.shape
    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    ky, kx = np.meshgrid(ky, kx)

    fftim = fft2(padded)
    phase_shift = np.exp(-2j*np.pi*(shift_y*ky + shift_x*kx))
    shifted_fft = fftim * phase_shift
    shifted_image = np.real(ifft2(shifted_fft))

    return (zoom(shifted_image[npad:-npad, npad:-npad], 1/oversample_factor))

def shift_image_fourier2(im0, shift_x, shift_y, oversample_factor = 2, npad = 10):
    from scipy.fft import fft2, ifft2, fftshift
    from scipy.ndimage import zoom, fourier_shift

    im = zoom(im0, oversample_factor)
    padded = np.pad(im, npad)

    # nx, ny = padded.shape
    # ky = np.fft.fftfreq(ny)
    # kx = np.fft.fftfreq(nx)
    # ky, kx = np.meshgrid(ky, kx)

    # fftim = fft2(padded)
    # phase_shift = np.exp(-2j*np.pi*(shift_y*ky + shift_x*kx))
    # shifted_fft = fftim * phase_shift
    shifted_image = np.real(ifft2(fourier_shift(fft2(padded), shift = (shift_y, shift_x))))

    return (zoom(shifted_image[npad:-npad, npad:-npad], 1/oversample_factor))

def parabolic_2d(coords, a, b, c, d, e, f):
    """2D parabolic function for fitting."""
    x, y = coords
    return a * x**2 + b * y**2 + c * x * y + d * x + e * y + f

def subpixel_centroid_2d(image, window_size=3):
    """
    Find the subpixel centroid of the peak by fitting a 2D parabola.

    Parameters:
    - image: 2D numpy array representing the image.
    - window_size: size of the square window around the peak pixel.

    Returns:
    - subpixel_centroid: tuple (y_centroid, x_centroid) with subpixel accuracy.
    """
    # Find the peak pixel (integer coordinates)
    peak_y, peak_x = np.unravel_index(np.argmax(image), image.shape)

    # Define a small window around the peak
    half_window = window_size // 2
    y_min, y_max = max(0, peak_y - half_window), min(image.shape[0], peak_y + half_window + 1)
    x_min, x_max = max(0, peak_x - half_window), min(image.shape[1], peak_x + half_window + 1)
    
    # Extract the window around the peak
    window = image[y_min:y_max, x_min:x_max]
    y_coords, x_coords = np.indices(window.shape)
    
    # Flatten the coordinates and intensity values for fitting
    x_coords = x_coords.ravel()
    y_coords = y_coords.ravel()
    intensities = window.ravel()

    # Fit the 2D parabola to the data in the window
    initial_guess = (1, 1, 0, 0, 0, np.max(window))
    popt, _ = curve_fit(parabolic_2d, (x_coords, y_coords), intensities, p0=initial_guess) #,
                        # bounds = ((-np.inf, -np.inf,  -np.inf, -np.inf, -np.inf, -np.inf),
                        #           (0, 0, np.inf, np.inf, np.inf, np.inf)))

    # Extract fitted parameters
    a, b, c, d, e, _ = popt

    # Calculate the vertex of the parabola (subpixel peak)
    # x_peak = (2*b*d - c*e) / (c**2 - 4*a*b)
    # y_peak = (2*a*e - c*d) / (c**2 - 4*a*b)
    denom = 4*a*b - c**2
    x_peak = (c*e - 2*b*d) / denom
    y_peak = (c*d - 2*a*e) / denom

    # Convert the subpixel coordinates back to the original image coordinates
    x_centroid = x_min + x_peak
    y_centroid = y_min + y_peak

    return (y_centroid, x_centroid)

def find_3point_peak(x, y):
    '''
    Find the peak by fitting parabola to three points (maximum and adjacent points)
    '''
    peakloc = np.argmax(y)

    assert (peakloc >= 1) and (peakloc <= len(x)-1), "peak is at the edge"

    xslice = x[np.array([peakloc - 1, peakloc, peakloc+1])]
    yslice = y[np.array([peakloc - 1, peakloc, peakloc+1])]
    coeffs = np.linalg.inv([[xslice[0]**2, xslice[0], 1],
                            [xslice[1]**2, xslice[1], 1],
                            [xslice[2]**2, xslice[2], 1]]) @ yslice
    return -coeffs[1] / 2 / coeffs[0]

def find_9point_center_of_mass(image):
    '''
    Find center of mass in 3x3 cutout image
    '''
    peak_y, peak_x = np.unravel_index(np.argmax(image), image.shape)

    y_min, y_max = peak_y - 1, peak_y + 2
    x_min, x_max = peak_x - 1, peak_x + 2
    
    assert (y_max <= image.shape[0]) and (y_min >= 0), "peak is at the edge (y)"
    assert (x_max <= image.shape[1]) and (x_min >= 0), "peak is at the edge (x)"

    image_slice = image[y_min:y_max, x_min:x_max]

    (cy, cx) = center_of_mass(image_slice)

    x_centroid = x_min + cx
    y_centroid = y_min + cy

    return (y_centroid, x_centroid)

def find_9point_peak_2d(image):
    '''
    Find the peak of 2D image by fitting 2D parabola to nine points (maximum and adjacent points)
    '''
    peak_y, peak_x = np.unravel_index(np.argmax(image), image.shape)

    y_min, y_max = peak_y - 1, peak_y + 2
    x_min, x_max = peak_x - 1, peak_x + 2
    
    assert (y_max <= image.shape[0]) and (y_min >= 0), "peak is at the edge (y)"
    assert (x_max <= image.shape[1]) and (x_min >= 0), "peak is at the edge (x)"

    image_slice = image[y_min:y_max, x_min:x_max]
    y_coords, x_coords = np.indices(image_slice.shape)

    x_coords = x_coords.ravel()
    y_coords = y_coords.ravel()
    intensities = image_slice.ravel()

    mat = np.array([np.array([x**2, y**2, x*y, x, y, 1]) for (x, y) in zip(x_coords, y_coords)])
    coeffs = np.linalg.pinv(mat) @ intensities

    denom = 4*coeffs[0]*coeffs[1] - coeffs[2]**2
    x_peak = (coeffs[2] * coeffs[4] - 2*coeffs[1] * coeffs[3]) / denom
    y_peak = (coeffs[2] * coeffs[3] - 2*coeffs[0] * coeffs[4]) / denom

    x_centroid = x_min + x_peak
    y_centroid = y_min + y_peak

    return (y_centroid, x_centroid)


def find_25point_peak_2d(image):
    '''
    Find the peak of 2D image by fitting 2D parabola to nine points (maximum and adjacent points)
    '''
    peak_y, peak_x = np.unravel_index(np.argmax(image), image.shape)

    y_min, y_max = peak_y - 2, peak_y + 3
    x_min, x_max = peak_x - 2, peak_x + 3
    
    assert (y_max <= image.shape[0]) and (y_min >= 0), "peak is at the edge (y)"
    assert (x_max <= image.shape[1]) and (x_min >= 0), "peak is at the edge (x)"

    image_slice = image[y_min:y_max, x_min:x_max]
    y_coords, x_coords = np.indices(image_slice.shape)

    x_coords = x_coords.ravel()
    y_coords = y_coords.ravel()
    intensities = image_slice.ravel()

    mat = np.array([np.array([x**2, y**2, x*y, x, y, 1]) for (x, y) in zip(x_coords, y_coords)])
    coeffs = np.linalg.pinv(mat) @ intensities

    denom = 4*coeffs[0]*coeffs[1] - coeffs[2]**2
    x_peak = (coeffs[2] * coeffs[4] - 2*coeffs[1] * coeffs[3]) / denom
    y_peak = (coeffs[2] * coeffs[3] - 2*coeffs[0] * coeffs[4]) / denom

    x_centroid = x_min + x_peak
    y_centroid = y_min + y_peak

    return (y_centroid, x_centroid)



def iter_find_blob(im, ini_x, ini_y, ini_width = 4, thres= 1e-2, maxval = 1, maxwidth = 8, plot = True, return_masked = False):

    '''
    Find blob from an image, around given (ini_x, ini_y) coordinate.

    '''
    import cv2

    truncated = True
    width_x0 = width_y0 = width_x1 = width_y1 = ini_width

    while ((truncated is True) and (max(width_x0, width_y0, width_x1, width_y1) <= maxwidth)):

        image = im[ini_y - width_y0: ini_y + width_y1,
                                    ini_x - width_x0: ini_x + width_x1].copy()
        image /= np.sum(image)

        _, binary = cv2.threshold(image, thres, maxval, cv2.THRESH_BINARY)  

        # connected component analysis
        labeled, num_features = label(binary)
        sizes = np.array([(labeled == i).sum() for i in range(1, num_features + 1)])
        
        # keep largest connected component
        largest_blob_label = 1 + np.argmax(sizes)  
        blob_mask = (labeled == largest_blob_label).astype(np.uint8)

        # if the mask touches any boundaries, extend the window
        rows, cols = blob_mask.shape
        if any(blob_mask[0,:]): width_y0 += 1
        elif any(blob_mask[:,0]): width_x0 += 1
        elif any(blob_mask[rows-1,:]): width_y1 += 1
        elif any(blob_mask[:,cols-1]): width_x1 += 1
        else:
            truncated = False

    if plot:
        plt.imshow(image)
        plt.contour(blob_mask, levels=[0.99], colors='white')
        plt.show()

    if return_masked: image *= blob_mask

    return image, ini_x - width_x0, ini_x + width_x1, ini_y - width_y0, ini_y + width_y1


def extract_patch(image, xcoor, ycoor, xwidth, ywidth, plot = False):

    '''
    Extract patch of the image centered at (xcoor, ycoor) with width (xwidth, ywidth).
    Shifts the image subpixel scale to sample image at (xcoor, ycoor) with subpixel accuracy.

    TODO WARNING: doesn't work on edges!! 
    '''
    import cv2

    _image = np.pad(image, (1, 1))
    xcoor += 1
    ycoor += 1

    dx = (xcoor - xwidth/2) - np.floor(xcoor - xwidth/2) #+ 0.5
    dy = (ycoor - ywidth/2) - np.floor(ycoor - ywidth/2) #+ 0.5

    if plot:
        plt.clf()
        plt.imshow(image, origin='lower')
        plt.plot((xcoor-1), (ycoor-1), 'x', color='red')
        plt.axvline((xcoor-1) - xwidth/2)
        plt.axvline((xcoor-1) + xwidth/2)
        plt.axhline((ycoor-1) - ywidth/2)
        plt.axhline((ycoor-1) + ywidth/2)

        print("shifting the image by -dx, -dy : %.3f, %.3f" % (-dx, -dy))

    M = np.float32([[1,0,-dx-0.5], [0,1,-dy-0.5]])
    shifted_image = cv2.warpAffine(_image, M, (_image.shape[1], _image.shape[0]))

    _ymin, _ymax, _xmin, _xmax = np.floor(ycoor - ywidth/2).astype(int), np.floor(ycoor - ywidth/2).astype(int) + ywidth, np.floor(xcoor - xwidth/2).astype(int), np.floor(xcoor - xwidth/2).astype(int) + xwidth
    
    assert (_ymax >= 0), "The requested patch is completely out of the canvas. ymax = %d < 0" % (_ymax)
    assert (_xmax >= 0), "The requested patch is completely out of the canvas. xmax = %d < 0" % (_xmax)
    assert (_xmin <= shifted_image.shape[1]), "The requested patch is completely out of the canvas. xmin = %d > %d" % (_xmin, shifted_image.shape[1])
    assert (_ymin <= shifted_image.shape[0]), "The requested patch is completely out of the canvas. ymin = %d > %d" % (_ymin, shifted_image.shape[0])

    print(_ymin, _ymax, _xmin, _xmax)
    print(np.shape(shifted_image))
    # patch = shifted_image[_ymin:_ymax, _xmin:_xmax]

    patch = shifted_image[max(0, _ymin): min(_ymax, shifted_image.shape[0]), 
                          max(0, _xmin): min(_xmax, shifted_image.shape[1])]
    
    print("expected patch size: %d x %d" % (ywidth, xwidth))
    print("acquired patch size: %d x %d" % (patch.shape[0], patch.shape[1]))

    if _ymin < 0:
        print('ymin', _ymin, np.shape(patch))
        d = -_ymin
        patch = np.vstack([np.zeros((d, np.shape(patch)[1])), patch])
        # patch = np.hstack([np.zeros((_image.shape[0], d)), patch])
    
    if _ymax > shifted_image.shape[0]:
        print('ymax', _ymax, np.shape(patch))
        d = _ymax - shifted_image.shape[0]
        patch = np.vstack([patch, np.zeros((d, np.shape(patch)[1]))])
        # patch = np.hstack([patch, np.zeros((_image.shape[0], d))])

    if _xmin < 0:
        print('xmin', _xmin, np.shape(patch))
        d = -_xmin
        # patch = np.vstack([np.zeros((d,_image.shape[1])), patch])
        patch = np.hstack([np.zeros((np.shape(patch)[0], d)), patch])
    
    if _xmax > shifted_image.shape[1]:
        print('xmax', _xmax, np.shape(patch))
        d = _xmax - shifted_image.shape[1]
        # patch = np.vstack([patch, np.zeros((d,_image.shape[1]))])
        patch = np.hstack([patch, np.zeros((np.shape(patch)[0], d))])

    print("final patch size: %d x %d" % (patch.shape[0], patch.shape[1]))

    return patch

def apply_patch_fft(canvas, patch, xcoor, ycoor):

    out_canvas = canvas.copy()

    _patch = np.pad(patch, (1,1))

def apply_patch(canvas, patch, xcoor, ycoor, shift_method = 'warpaffine', plot = False, verbose = False):

    '''
    Apply patch to the canvas centered at (xcoor, ycoor).
    Shifts the patch subpixel scale for subpixel accuracy.

    '''
    import cv2

    # assert (xcoor > 1) and (ycoor > 1)
    out_canvas = canvas.copy()

    _patch = np.pad(patch, (1,1))

    canvas_height, canvas_width = canvas.shape
    height, width = _patch.shape
    xp, yp = width/2, height/2

    dx = (xcoor - xp) - np.floor(xcoor - xp) #+ 0.5
    dy = (ycoor - yp) - np.floor(ycoor - yp) #+ 0.5
    # print("shifting the image by dx, dy : %.3f, %.3f" % (dx, dy))
    # print("actually shifting the image by dx, dy : %.3f, %.3f" % (dx+0.5, dy+0.5))
    # extend the patch for subpixel shifting
    extended_patch = (np.hstack([np.vstack([_patch, np.zeros(np.shape(_patch)[1])]), np.zeros((np.shape(_patch)[0]+1,1))]))

    # M = np.float32([[1,0,dx+0.5], [0,1,dy+0.5]])
    # shifted_patch = cv2.warpAffine(extended_patch, M, (extended_patch.shape[1], extended_patch.shape[0]))

    # shift the image
    # if shift_method == 'warpaffine':
    shifted_patch = shift_image_warpaffine(extended_patch, dx+0.5, dy+0.5)
    # elif shift_method == 'fft':
    #     shifted_patch = shift_image_fourier(extended_patch, dx+0.5, dy+0.5, oversample_factor = 2, npad = 10)
    # else:
    #     raise ValueError("shift_method should be either 'warpaffine' or 'fft'")
        
    
    _ymin, _ymax, _xmin, _xmax = np.floor(ycoor - yp).astype(int), np.floor(ycoor - yp).astype(int)+height+1,np.floor(xcoor - xp).astype(int),np.floor(xcoor - xp).astype(int)+width+1
    
    if verbose:
        print("pasting the (%s) shifted patch to the coordinates" % (str(np.shape(shifted_patch))),
          "ymin, ymax, xmin, xmax", _ymin, _ymax, _xmin, _xmax,
          "yrange, xrange", canvas_height, canvas_width, 
          )
    
    if _ymin < 0:
        if verbose:print("trimming first %d index of y" % (-_ymin))
        shifted_patch = shifted_patch[-_ymin:]
        _ymin = 0
    
    if _ymax > canvas_height:
        if verbose:print("trimming last %d index of y" % (_ymax-canvas_height))
        shifted_patch = shifted_patch[:-(_ymax-canvas_height)]
        _ymax = canvas_height
    
    if _xmin < 0:
        if verbose:print("trimming first %d index of x" % (-_xmin))
        shifted_patch = shifted_patch[:,-_xmin:]
        _xmin = 0

    if _xmax > canvas_width:
        if verbose:print("trimming last %d index of x" % (_xmax-canvas_width))
        shifted_patch = shifted_patch[:,:-(_xmax-canvas_width)]
        _xmax = canvas_width

    
    out_canvas[_ymin:_ymax, _xmin:_xmax] += shifted_patch #[:-1, :-1]
    
    if plot:
        plt.clf()
        print("shifting the image by dx, dy : %.3f, %.3f" % (dx, dy))
        plt.imshow(canvas, origin='lower')
        plt.plot(xcoor, ycoor, 'x', color='red')
        plt.axvline(xcoor - (width-2)/2)
        plt.axvline(xcoor + (width-2)/2)
        plt.axhline(ycoor - (height-2)/2)
        plt.axhline(ycoor + (height-2)/2)

    return out_canvas #, shifted_patch



def extract_patch_oversampled(image, xcoor, ycoor, xwidth, ywidth, oversample_factor, plot = False, return_downsampled = False):

    '''
    Extract patch of the image centered at (xcoor, ycoor) with width (xwidth, ywidth).
    Shifts the image subpixel scale to sample image at (xcoor, ycoor) with subpixel accuracy.

    TODO WARNING: doesn't work on edges!! 
    '''
    import cv2


    oversampled_image = cv2.resize(
        image, 
        None, 
        fx=oversample_factor, 
        fy=oversample_factor, 
        interpolation=cv2.INTER_LINEAR
    )

    xcoor_oversampled = xcoor * oversample_factor
    ycoor_oversampled = ycoor * oversample_factor
    xwidth_oversampled = int(xwidth * oversample_factor)
    ywidth_oversampled = int(ywidth * oversample_factor)


    _image = np.pad(oversampled_image, (1, 1))
    xcoor_oversampled += 1
    ycoor_oversampled += 1

    dx = (xcoor_oversampled - xwidth_oversampled/2) - np.floor(xcoor_oversampled - xwidth_oversampled/2) #+ 0.5
    dy = (ycoor_oversampled - ywidth_oversampled/2) - np.floor(ycoor_oversampled - ywidth_oversampled/2) #+ 0.5

    if plot:
        plt.clf()
        plt.imshow(oversampled_image, origin='lower')
        plt.plot((xcoor_oversampled-1), (ycoor_oversampled-1), 'x', color='red')
        plt.axvline((xcoor_oversampled-1) - xwidth_oversampled/2)
        plt.axvline((xcoor_oversampled-1) + xwidth_oversampled/2)
        plt.axhline((ycoor_oversampled-1) - ywidth_oversampled/2)
        plt.axhline((ycoor_oversampled-1) + ywidth_oversampled/2)

        print("shifting the image by -dx, -dy : %.3f, %.3f" % (-dx, -dy))

    M = np.float32([[1,0,-dx-0.5], [0,1,-dy-0.5]])
    shifted_image = cv2.warpAffine(_image, M, (_image.shape[1], _image.shape[0]))

    _ymin, _ymax, _xmin, _xmax = np.floor(ycoor_oversampled - ywidth_oversampled/2).astype(int), np.floor(ycoor_oversampled - ywidth_oversampled/2).astype(int) + ywidth_oversampled, np.floor(xcoor_oversampled - xwidth_oversampled/2).astype(int), np.floor(xcoor_oversampled - xwidth_oversampled/2).astype(int) + xwidth_oversampled
    
    assert (_ymax >= 0), "The requested patch is completely out of the canvas. ymax = %d < 0" % (_ymax)
    assert (_xmax >= 0), "The requested patch is completely out of the canvas. xmax = %d < 0" % (_xmax)
    assert (_xmin <= shifted_image.shape[1]), "The requested patch is completely out of the canvas. xmin = %d > %d" % (_xmin, shifted_image.shape[1])
    assert (_ymin <= shifted_image.shape[0]), "The requested patch is completely out of the canvas. ymin = %d > %d" % (_ymin, shifted_image.shape[0])

    print(_ymin, _ymax, _xmin, _xmax)
    print(np.shape(shifted_image))
    # patch = shifted_image[_ymin:_ymax, _xmin:_xmax]

    patch = shifted_image[max(0, _ymin): min(_ymax, shifted_image.shape[0]), 
                          max(0, _xmin): min(_xmax, shifted_image.shape[1])]
    
    print("expected patch size: %d x %d" % (ywidth, xwidth))
    print("acquired patch size: %d x %d" % (patch.shape[0], patch.shape[1]))

    if _ymin < 0:
        print('ymin', _ymin, np.shape(patch))
        d = -_ymin
        patch = np.vstack([np.zeros((d, np.shape(patch)[1])), patch])
        # patch = np.hstack([np.zeros((_image.shape[0], d)), patch])
    
    if _ymax > shifted_image.shape[0]:
        print('ymax', _ymax, np.shape(patch))
        d = _ymax - shifted_image.shape[0]
        patch = np.vstack([patch, np.zeros((d, np.shape(patch)[1]))])
        # patch = np.hstack([patch, np.zeros((_image.shape[0], d))])

    if _xmin < 0:
        print('xmin', _xmin, np.shape(patch))
        d = -_xmin
        # patch = np.vstack([np.zeros((d,_image.shape[1])), patch])
        patch = np.hstack([np.zeros((np.shape(patch)[0], d)), patch])
    
    if _xmax > shifted_image.shape[1]:
        print('xmax', _xmax, np.shape(patch))
        d = _xmax - shifted_image.shape[1]
        # patch = np.vstack([patch, np.zeros((d,_image.shape[1]))])
        patch = np.hstack([patch, np.zeros((np.shape(patch)[0], d))])

    print("final patch size: %d x %d" % (patch.shape[0], patch.shape[1]))

    if return_downsampled:
        patch_downsampled = cv2.resize(patch, 
                               (xwidth, ywidth),  # (width, height) in original scale
                               interpolation=cv2.INTER_LINEAR)
        return patch_downsampled
    return patch



def dist(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_centroid(image, thres = 1, verbose= False):

    '''
    iterate through a few centroid finding algorithms until it gives a good result
    '''

    func_names = ['subpixel_centroid_2d', 'find_9point_peak_2d', 'find_25point_peak_2d', 'center_of_mass', 'find_9point_center_of_mass']
    functions = [subpixel_centroid_2d, find_9point_peak_2d, find_25point_peak_2d, center_of_mass, find_9point_center_of_mass]
    maxloc_y, maxloc_x = np.unravel_index(np.argmax(image), image.shape)

    for (func, fname) in zip(functions, func_names):
        (cy, cx) = func(image)

        if dist((cx, cy), (maxloc_x, maxloc_y)) < thres:

            if verbose:
                print("function %s successful" % fname)
            
            return (cy, cx), fname
    if verbose: print("no function was successful")
    return (cy, cx), fname


def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def make_flat_image(image, cutoff = 200, mincut = 0.5, maxcut = 1.5):

    image_filtered = np.array([highpass_filter(im1d, cutoff, len(im1d)) for im1d in image])
    flat =  image / (image - image_filtered)
    flat[flat < mincut] = 1
    flat[flat > maxcut] = 1
    return flat