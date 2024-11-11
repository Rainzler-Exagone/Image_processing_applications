import cv2
import matplotlib.pyplot as plt
import numpy as np

#charger une image
image = cv2.imread('<your_image>')

# Converion en RGB pour affichage ave matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


#Affichage
plt.imshow(image_rgb)
plt.axis('off')
plt.show()


#Converion en RGB pour affichage ave matplotlib
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#Affichage
plt.imshow(gray_image,cmap='gray')
plt.axis('off')
plt.show()


def adjust_brightness_contrast(image, brightness=0,contrast=0):
    return cv2.convertScaleAbs(image,alpha=1+contrast/100 , beta=brightness)

image_adjusted = adjust_brightness_contrast(image, brightness=10 , contrast=60)
plt.imshow(cv2.cvtColor(image_adjusted, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


#histogram
plt.hist(gray_image.ravel(), bins=8, range= [0,256])
plt.title('Histogramme de limage de niveaux gris')
plt.xlabel('Intensity of pixels')
plt.ylabel('Number of pixels')
plt.show()

#blurred
blurred_image = cv2.GaussianBlur(image,(15,15),0)
plt.imshow(cv2.cvtColor(blurred_image,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


sobelx = cv2.Sobel(gray_image,cv2.CV_64F,1,0, ksize=5)
sobely = cv2.Sobel(gray_image,cv2.CV_64F,0,1, ksize=5)

abs_sobelx = cv2.convertScaleAbs(sobelx)
abs_sobely = cv2.convertScaleAbs(sobely)

# sobel  = 
sobel  = cv2.addWeighted(abs_sobelx, 0.5,abs_sobely , 0.5, 0)


plt.subplot(1,2,1),plt.imshow(sobelx,cmap='gray')
plt.title('Sobel X')
plt.axis('off')

plt.subplot(1,2,2),plt.imshow(sobely,cmap='gray')
plt.title('Sobel Y')
plt.axis('off')

plt.show()
plt.imshow(sobel, cmap='gray')
plt.title('Sobel')
plt.axis('off')
plt.show()

#edges
edges = cv2.Canny(gray_image,100,300)
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()


# #rotation
# rows, cols = gray_image.shape
# rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 75, 1)
# rotated_image = cv2.warpAffine(gray_image,rotation_matrix,(cols,rows))

# #Affichage
# plt.imshow(rotated_image,cmap='gray')
# plt.axis('off')
# plt.title('Image rotated')
# plt.show()

dft =cv2.dft(np.float32(gray_image),flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))


plt.imshow(magnitude_spectrum,cmap='gray')
plt.title('Magnitude spectrum')
plt.axis('off')
plt.show()


#binary image
_, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(binary_image, cmap='gray')
plt.axis('off')
plt.title('Binary image')
plt.show()
#erosion 

kernel = np.ones((5, 5), np.uint8)
eroded_image = cv2.erode(binary_image, kernel, iterations=1)
dilated_image = cv2.dilate(binary_image, kernel, iterations=1) 

plt.subplot(1,2,1),
plt.imshow(eroded_image, cmap='gray')
plt.title('Eroded image')
plt.axis('off')
plt.subplot(1, 2, 2),
plt.imshow(dilated_image, cmap='gray')
plt.title('Dilated image')
plt.axis('off')
plt.show()


#
def apply_opening(image, kernel_size=5, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
     
def apply_closing(image, kernel_size=5, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
     

#Apply opening and closing
opened_image = apply_opening(binary_image)
closed_image = apply_closing(binary_image)

#Display
plt.figure(figsize=(12,6))
plt.subplot(1, 3, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary image')
plt.axis('off')


plt.subplot(1, 3, 2)
plt.imshow(opened_image, cmap='gray')
plt.title('Opened image (Remove Noise)')
plt.axis('off')


plt.subplot(1, 3, 3)
plt.imshow(closed_image, cmap='gray')
plt.title('Closed image (Closes Gaps)')
plt.axis('off')

plt.show()



#
def apply_sharpening(gray_image):
    kernel = np.array([[0,-1,0],
                       [-1, 5,-1],
                       [0,-1,0]])
    return cv2.filter2D(gray_image, -1, kernel)

#example usage
sharpened_image = apply_sharpening(gray_image)
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(sharpened_image, cmap='gray')
plt.title("Sharpened Image")
plt.axis('off')
plt.show()

#equalize
equalized_image = cv2.equalizeHist(gray_image)
plt.imshow(equalized_image, cmap='gray')
plt.title('Histogramme equalize image')
plt.axis('off')
plt.show()


plt.subplot(1, 2, 1)
plt.hist(gray_image.ravel(), bins=8, range= [0, 256])
plt.title('Histogramme de l\'image gris')
plt.xlabel('Intensity of pixels')
plt.ylabel('Number of pixels')

plt.subplot(1, 2, 2)
plt.hist(equalized_image.ravel(), bins=8, range= [0, 256])
plt.title('Histogramme de l\'image equalizé gris')
plt.xlabel('Intensity of pixels')
plt.ylabel('Number of pixels')


plt.show()
