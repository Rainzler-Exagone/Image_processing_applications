import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('brain.jpg',0)

#add noise
mean = 0
sigma = 20 
gaussian_noise = np.random.normal(mean, sigma, image.shape)
noisy_image = cv2.add(image,gaussian_noise.astype(np.uint8))



dft = cv2.dft(np.float32(noisy_image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

#Afficher l'image l'origine
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1),plt.imshow(image,cmap='gray')
plt.title('Original Image'), plt.axis('off')
plt.subplot(1,2,2), plt.imshow(noisy_image, cmap='gray')



#creer

rows, cols = noisy_image.shape
crow , ccol = rows //2 , cols //2 #Centre de l'image
mask = np.zeros((rows, cols,2),np.uint8)
mask[crow-30:crow+30 ,ccol-30:ccol+30] = 1  #Taille du filtre

#Appliquer  le mask au domain frequential
filtered_dft = dft_shifted * mask

dft_ishift = np.fft.ifftshift(filtered_dft)
image_filtered = cv2.idft(dft_ishift)
image_filtered = cv2.magnitude(image_filtered[:,:,0],image_filtered[:,:,1])

#Afficher l'image filtré 
plt.figure(figsize=(12,6))
plt.subplot(1,2,1), plt.imshow(noisy_image,cmap='gray')
plt.title("Image with noise")
plt.axis('off')
plt.subplot(1, 2, 2), plt.imshow(image_filtered, cmap='gray')
plt.title("Filtered Image")
plt.show()