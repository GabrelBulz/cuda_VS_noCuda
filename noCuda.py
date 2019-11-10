import numpy as np 
import numba 
from numba import njit, config, threading_layer, prange
import cv2
from PIL import Image
import datetime

img_name="eu.jpg"



# config.THREADING_LAYER = 'default'


"""
Using parallel we can access the fusion loop as in the example below
Parallel loop listing for  Function test, example.py (4)
--------------------------------------|loop #ID
@njit(parallel=True)                  |
def test(x):                          |
    n = x.shape[0]                    |
    a = np.sin(x)---------------------| #0
    b = np.cos(a * a)-----------------| #1
    acc = 0                           |
    for i in prange(n - 2):-----------| #3
        for j in prange(n - 1):-------| #2
            acc += b[i] + b[j + 1]    |
    return acc                        |


    Parallel region 0:
+--0 (parallel, fused with loop(s): 1)


Parallel region 1:
+--3 (parallel)
+--2 (serial)
"""

def paralellize_to_gray_each_row(img, gray_buff, size):
	for i in prange(size[0]):
		gray_buff[i] = to_gray_row(img[i])

	return gray_buff, np.shape(gray_buff)


def to_gray_row(row):

	new_gray_row = []

	for i in prange(np.shape(row)[0]):
		red = row[i][0]
		green = row[i][1]
		blue = row[i][2]

		new_gray_row.append(0.2125*red + 0.7154*green + 0.0721*blue)

	return new_gray_row

def copy_matrixx(matrix, size, center_row, center_col):
	"""
		extract a submatrix of a centrain size from 
		a bigger one given as param matrxi
	"""
	offset = -(size//2)

	temp = [[0,0,0], [0,0,0], [0,0,0]]
	for i in range(3):
		for j in range(3):
			temp[i][j] = matrix[offset+i][offset+j]

	return temp


def gaussian_blur_prepare_parallel(img,size):
	"""
		make changes in temp matrix becuase
		we will affect the value as we go
	"""
	img_result = img.copy()

	for i in range (1, size[0]-1):
		for j in range (1, size[1]-1):
			submat = copy_matrixx(img,3,i,j)
			img_result[i][j] = gaussian_blur(submat,3)

	return img_result


def gaussian_blur(frame, size):
	kernel_gaussian_blur = [[1,2,1],
							  [2,4,2],
						      [1,2,1]]
	gaussian_value = 16
	sum = 0

	for i in prange(size):
		for j in prange(size):
			sum = sum + frame[i][j] * kernel_gaussian_blur[i][j]

	return sum/gaussian_value

def write_img(img, name):
	out_img = Image.fromarray(img)
	out_img.convert('RGB').save(name, "PNG", optimize=True)


start = datetime.datetime.now()

img_desc = Image.open(img_name)
img_desc.load()

img = np.asarray(img_desc, dtype="float64")
size = np.shape(img)

gray = np.zeros([size[0], size[1]], dtype="float64")

gray, gray_size = paralellize_to_gray_each_row(img, gray, size)
write_img(gray, 'ceva.png')
gray_gaussian = gaussian_blur_prepare_parallel(gray, gray_size)
write_img(gray_gaussian, "gray1_gauss.png")

finish = datetime.datetime.now()
print("it took " + str(finish-start) + " for the CPU to finish the job")




