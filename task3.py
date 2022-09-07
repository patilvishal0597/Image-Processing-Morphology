"""
Morphology Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with commonly used morphology
binary image processing techniques. Use the proper combination of the four commonly used morphology operations, 
i.e. erosion, dilation, open and close, to remove noises and extract boundary of a binary image. 
Specifically, you are given a binary image with noises for your testing, which is named 'task3.png'.  
Note that different binary image might be used when grading your code. 

You are required to write programs to: 
(i) implement four commonly used morphology operations: erosion, dilation, open and close. 
    The stucturing element (SE) should be a 3x3 square of all 1's for all the operations.
(ii) remove noises in task3.png using proper combination of the above morphology operations. 
(iii) extract the boundaries of the objects in denoised binary image 
      using proper combination of the above morphology operations. 
Hint: 
• Zero-padding is needed before morphology operations. 

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy libraries, HOWEVER, 
you are NOT allowed to use any functions or APIs directly related to morphology operations.
Please implement erosion, dilation, open and close operations ON YOUR OWN.
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np

struct_element = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(int)

def morph_erode(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return erode_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology erosion on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    m,n = np.shape(img)
    img = 1 * (img == 255)
    padded_img = np.pad(img, (1,), 'constant')
    erode_img = np.zeros((m,n))
    for i in range(1, m+1):
        for j in range(1, n+1):
            erosion_flag = True
            for p in range(3):
                for q in range(3):
                    if struct_element[p][q] != padded_img[i + p - 1][j + q - 1]:
                        erosion_flag = False
                        break
            
            if erosion_flag == True:
                erode_img[i-1][j-1] = 1
            else:
                erode_img[i-1][j-1] = 0
                
    erode_img = 255*erode_img
    return erode_img

def morph_dilate(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return dilate_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology dilation on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """

    # TO DO: implement your solution here
    m,n = np.shape(img)
    img = 1 * (img == 255)
    padded_img = np.pad(img, (1,), 'constant')
    dilate_img = np.zeros((m,n))
    for i in range(1, m+1):
        for j in range(1, n+1):
            dilation_flag = False
            for p in range(3):
                for q in range(3):
                    if struct_element[p][q] == padded_img[i + p - 1][j + q - 1]:
                        dilation_flag = True
                        break
            
            if dilation_flag == True:
                dilate_img[i-1][j-1] = 1
            else:
                dilate_img[i-1][j-1] = 0
                
    dilate_img = 255*dilate_img
    return dilate_img

def morph_open(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return open_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology opening on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """

    # TO DO: implement your solution here
    
    open_img = morph_dilate(morph_erode(img))
    return open_img

def morph_close(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return close_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology closing on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """

    # TO DO: implement your solution here
    
    close_img = morph_erode(morph_dilate(img))
    return close_img

def denoise(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Remove noises from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    return morph_close(morph_open(img))

def boundary(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Extract boundaries from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """

    # TO DO: implement your solution here
    bound_img = img - morph_erode(img)
    return bound_img

if __name__ == "__main__":
    img = imread('task3.png', IMREAD_GRAYSCALE)
    denoise_img = denoise(img)
    imwrite('results/task3_denoise.jpg', denoise_img)
    bound_img = boundary(denoise_img)
    imwrite('results/task3_boundary.jpg', bound_img)