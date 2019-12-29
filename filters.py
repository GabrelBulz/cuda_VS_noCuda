import numpy as np 
import numba 
from numba import njit, config, threading_layer, prange
import cv2
from PIL import Image
import datetime

img_name="eu.jpg"



# config.THREADING_LAYER = 'default'


###########################################################################################################################################
@njit(parallel=True)
def paralellize_to_gray_each_row(img, gray_buff, gray_row_buff, size):
	for i in prange(size[0]):

		to_gray_row(img[i], gray_buff, i)
		# x = gray_row_buff.copy()
		# print(gary_row_buff)
		# print(x)
		# gray_buff[i]= temp
		# print(gary_row_buff[0])
		# gray_buff[i][0] = 55

		# print(gray_buff[i])
		# test_print(gary_row_buff)

		# print(gray_buff)

	# return gray_buff, np.shape(gray_buff)

@njit(parallel=True)
def to_gray_row(row, gray_buff, curr_line):

	for i in prange(size[1]):
		red = row[i][0]
		green = row[i][1]
		blue = row[i][2]

		gray_buff[curr_line][i]=(0.2125*red + 0.7154*green + 0.0721*blue)


@njit(parallel=True)
def to_gray_parallel(img, gray_buff, rows, cols):
	for i in prange(rows//5):
		for j in prange(cols//5):
			rgb1 = img[i*5][j*5]
			rgb2 = img[i*5][(j*5)+1]
			rgb3 = img[i*5][(j*5)+2]
			rgb4 = img[i*5][(j*5)+3]
			rgb5 = img[i*5][(j*5)+4]

			rgb6 = img[(i*5)+1][(j*5)]
			rgb7 = img[(i*5)+1][(j*5)+1]
			rgb8 = img[(i*5)+1][(j*5)+2]
			rgb9 = img[(i*5)+1][(j*5)+3]
			rgb10 = img[(i*5)+1][(j*5)+4]

			rgb11 = img[(i*5)+2][(j*5)]
			rgb12 = img[(i*5)+2][(j*5)+1]
			rgb13 = img[(i*5)+2][(j*5)+2]
			rgb14 = img[(i*5)+2][(j*5)+3]
			rgb15 = img[(i*5)+2][(j*5)+4]

			rgb16 = img[(i*5)+3][(j*5)]
			rgb17 = img[(i*5)+3][(j*5)+1]
			rgb18 = img[(i*5)+3][(j*5)+2]
			rgb19 = img[(i*5)+3][(j*5)+3]
			rgb20 = img[(i*5)+3][(j*5)+4]

			rgb21 = img[(i*5)+4][(j*5)]
			rgb22 = img[(i*5)+4][(j*5)+1]
			rgb23 = img[(i*5)+4][(j*5)+2]
			rgb24 = img[(i*5)+4][(j*5)+3]
			rgb25 = img[(i*5)+4][(j*5)+4]



			gray_buff[i*5][j*5] = 0.2125*rgb1[0] + 0.7154*rgb1[1] + 0.0721*rgb1[2]
			gray_buff[i*5][(j*5)+1] = 0.2125*rgb2[0] + 0.7154*rgb2[1] + 0.0721*rgb2[2]
			gray_buff[i*5][(j*5)+2] = 0.2125*rgb3[0] + 0.7154*rgb3[1] + 0.0721*rgb3[2]
			gray_buff[i*5][(j*5)+3] = 0.2125*rgb4[0] + 0.7154*rgb4[1] + 0.0721*rgb4[2]
			gray_buff[i*5][(j*5)+4] = 0.2125*rgb5[0] + 0.7154*rgb5[1] + 0.0721*rgb5[2]

			gray_buff[(i*5)+1][(j*5)] = 0.2125*rgb6[0] + 0.7154*rgb6[1] + 0.0721*rgb6[2]
			gray_buff[(i*5)+1][(j*5)+1] = 0.2125*rgb7[0] + 0.7154*rgb7[1] + 0.0721*rgb7[2]
			gray_buff[(i*5)+1][(j*5)+2] = 0.2125*rgb8[0] + 0.7154*rgb8[1] + 0.0721*rgb8[2]
			gray_buff[(i*5)+1][(j*5)+3] = 0.2125*rgb9[0] + 0.7154*rgb9[1] + 0.0721*rgb9[2]
			gray_buff[(i*5)+1][(j*5)+4] = 0.2125*rgb10[0] + 0.7154*rgb10[1] + 0.0721*rgb10[2]

			gray_buff[(i*5)+2][(j*5)] = 0.2125*rgb11[0] + 0.7154*rgb11[1] + 0.0721*rgb11[2]
			gray_buff[(i*5)+2][(j*5)+1] = 0.2125*rgb12[0] + 0.7154*rgb12[1] + 0.0721*rgb12[2]
			gray_buff[(i*5)+2][(j*5)+2]= 0.2125*rgb13[0] + 0.7154*rgb13[1] + 0.0721*rgb13[2]
			gray_buff[(i*5)+2][(j*5)+3] = 0.2125*rgb14[0] + 0.7154*rgb14[1] + 0.0721*rgb14[2]
			gray_buff[(i*5)+2][(j*5)+4] = 0.2125*rgb15[0] + 0.7154*rgb15[1] + 0.0721*rgb15[2]

			gray_buff[(i*5)+3][(j*5)] = 0.2125*rgb16[0] + 0.7154*rgb16[1] + 0.0721*rgb16[2]
			gray_buff[(i*5)+3][(j*5)+1]= 0.2125*rgb17[0] + 0.7154*rgb17[1] + 0.0721*rgb17[2]
			gray_buff[(i*5)+3][(j*5)+2] = 0.2125*rgb18[0] + 0.7154*rgb18[1] + 0.0721*rgb18[2]
			gray_buff[(i*5)+3][(j*5)+3] = 0.2125*rgb19[0] + 0.7154*rgb19[1] + 0.0721*rgb19[2]
			gray_buff[(i*5)+3][(j*5)+4] = 0.2125*rgb20[0] + 0.7154*rgb20[1] + 0.0721*rgb20[2]

			gray_buff[(i*5)+4][(j*5)] = 0.2125*rgb21[0] + 0.7154*rgb21[1] + 0.0721*rgb21[2]
			gray_buff[(i*5)+4][(j*5)+1] = 0.2125*rgb22[0] + 0.7154*rgb22[1] + 0.0721*rgb22[2]
			gray_buff[(i*5)+4][(j*5)+2] = 0.2125*rgb23[0] + 0.7154*rgb23[1] + 0.0721*rgb23[2]
			gray_buff[(i*5)+4][(j*5)+3] = 0.2125*rgb24[0] + 0.7154*rgb24[1] + 0.0721*rgb24[2]
			gray_buff[(i*5)+4][(j*5)+4] = 0.2125*rgb25[0] + 0.7154*rgb25[1] + 0.0721*rgb25[2]



@njit(parallel=True)
def to_gray_parallel_kernel2(img, gray_buff, rows, cols):
	for i in prange(rows//2):
		for j in prange(cols//2):
			rgb1 = img[i*2][j*2]
			rgb2 = img[i*2][(j*2)+1]
			rgb3 = img[(i*2)+1][(j*2)]
			rgb4 = img[(i*2)+1][(j*2)+1]

			gray_buff[i*2][j*2] = 0.2125*rgb1[0] + 0.7154*rgb1[1] + 0.0721*rgb1[2]
			gray_buff[i*2][(j*2)+1] = 0.2125*rgb2[0] + 0.7154*rgb2[1] + 0.0721*rgb2[2]
			gray_buff[(i*2)+1][(j*2)] = 0.2125*rgb3[0] + 0.7154*rgb3[1] + 0.0721*rgb3[2]
			gray_buff[(i*2)+1][(j*2)+1] = 0.2125*rgb4[0] + 0.7154*rgb4[1] + 0.0721*rgb4[2]


@njit(parallel=True)
def test1(img, gray_buff, rows, cols):
	for i in prange(rows//5):
		for j in prange(cols//5):
			test2(img, gray_buff, rows, cols, i, j)


@njit(parallel=True)
def test2(img, gray_buff, rows, cols, x, y):
	for i in prange(5):
		for j in prange(5):
			rgb = img[(x*5)+i][(y*5)+j]
			gray_buff[(x*5)+i][(y*5)+j] = 0.2125*rgb[0] + 0.7154*rgb[1] + 0.0721*rgb[2]


@njit(parallel=True)
def rbg_to_gray(rows, cols, dest, src):
	for i in prange(rows):
		for j in prange(cols):
			rgb = src[i][j]
			dest[i][j] = 0.2125*rgb[0] + 0.7154*rgb[1] + 0.0721*rgb[2]

	return dest


######################################################################################################################################################################3




# @njit
# def copy_matrixx(matrix, size, center_row, center_col):
# 	"""
# 		extract a submatrix of a centrain size from 
# 		a bigger one given as param matrxi
# 	"""
# 	offset = -(size//2)

# 	temp = [[0,0,0], [0,0,0], [0,0,0]]
# 	for i in prange(3):
# 		for j in prange(3):
# 			temp[i][j] = matrix[offset+i][offset+j]

# 	return temp

# @njit
# def gaussian_blur_prepare_parallel(img,size):
# 	"""
# 		make changes in temp matrix becuase
# 		we will affect the value as we go
# 	"""
# 	img_result = img.copy()

# 	for i in prange (1, size[0]-1):
# 		for j in prange (1, size[1]-1):
# 			submat = copy_matrixx(img,3,i,j)
# 			img_result[i][j] = gaussian_blur(submat,3)

# 	return img_result

# @njit
# def gaussian_blur(frame, size):
# 	kernel_gaussian_blur = [[1,2,1],
# 							  [2,4,2],
# 						      [1,2,1]]
# 	gaussian_value = 16
# 	sum = 0

# 	for i in prange(size):
# 		for j in prange(size):
# 			sum = sum + frame[i][j] * kernel_gaussian_blur[i][j]

# 	return sum/gaussian_value

@njit(parallel=True)
def gaussian(img, gauss_kernel, rows, cols):
	for i in prange(1, rows-1):
		for j in prange(1, cols-1):

			gauss1 = img[i-1][j-1] * gauss_kernel[0][0]
			gauss2 = img[i-1][j  ] * gauss_kernel[0][1]
			gauss3 = img[i-1][j+1] * gauss_kernel[0][2]

			gauss4 = img[i][j-1] * gauss_kernel[1][0]
			gauss5 = img[i][j  ] * gauss_kernel[1][1]
			gauss6 = img[i][j+1] * gauss_kernel[1][2]

			gauss7 = img[i+1][j-1] * gauss_kernel[2][0]
			gauss8 = img[i+1][j  ] * gauss_kernel[2][1]
			gauss9 = img[i+1][j+1] * gauss_kernel[2][2]

			img[i][j] = (gauss1 + gauss2 + gauss3 + gauss4 + gauss5 + gauss6 + gauss6 + gauss7 + gauss8 + gauss9) / 16


def write_img(img, name):
	out_img = Image.fromarray(img)
	out_img.convert('RGB').save(name, "PNG", optimize=True)



start = datetime.datetime.now()

# xx = np.zeros(200, dtype="float64")
# test_np(xx)
# print(xx)


img_desc = Image.open(img_name)
img_desc.load()

img = np.asarray(img_desc, dtype="float64")
size = np.shape(img)

gauss_kernel = np.zeros((3,3), dtype="float64")
gauss_kernel[0][0] = 1
gauss_kernel[0][1] = 2 
gauss_kernel[0][2] = 1

gauss_kernel[1][0] = 2
gauss_kernel[1][1] = 4
gauss_kernel[1][2] = 2

gauss_kernel[2][0] = 1 
gauss_kernel[2][1] = 2
gauss_kernel[2][2] = 1

# print(gauss_kernel)

gray = np.zeros([size[0], size[1]], dtype="float64")
gray_row_buff = np.zeros(size[1], dtype="float64")


# paralellize_to_gray_each_row(img, gray, gray_row_buff, size)
to_gray_parallel_kernel2(img,gray,size[0], size[1])

gaussian(gray, gauss_kernel, size[0], size[1])

# to_gray_parallel(img,gray,size[0], size[1])
# test1(img,gray,size[0], size[1])

finish = datetime.datetime.now()
print("it took " + str(finish-start) + " for the gpu to finish the job")

write_img(gray, 'final_result.png')





