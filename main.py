import sys
import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.image as mpimg
import mysubroutines
from skimage.metrics import structural_similarity as ssim

argc = len(sys.argv)

image_path = sys.argv[1] if argc>1 else "image/peppers256.png"  # path of true image
lambda_weight = float(sys.argv[2]) if argc>2 else 0.001   # lambda coefficient, 0.04
mu_weight = float(sys.argv[3]) if argc>3 else 0.009   # mu coefficient
output_image = (sys.argv[4] if argc>4 else "result/fig1_result.png")   # path of restoration result
noise = float(sys.argv[5]) if argc>5 else 0.01   # noise level
print('Path of image_true:',image_path)
print('Path of result:', output_image)
print('Noise level=', noise)
print('lambda=%f, mu=%f' %(lambda_weight, mu_weight))

# setting some coefficients
delta = 1   # step of ADMM
tol_cg = 1e-8   # tolerance of CG iteration
tol_iter = 1e-6   # tolerance of ADMM iteration
max_cg = 1000   # maximum CG iteration times
max_iter = 1000   # maximum ADMM iteration times
kernel_size = 15   # size of the Gaussian kernel
gaussian_sigma = 1   # sigma of the Gaussian kernel

# Restoration
image_true = mpimg.imread(image_path)   # true image
d = len(image_true)   # dimension of image
kernel = mysubroutines.gaussian_kernel(kernel_size, gaussian_sigma)   # Gaussian kernel
np.random.seed(6)
noiseim = noise * np.random.randn(d, d)*np.max(image_true)   # noise signal
image_damage = mysubroutines.convolution(image_true, kernel)+noiseim   # image damaged by Gaussian blur and Gaussian noise

result = mysubroutines.ADMM(image_true, kernel, image_damage, mu_weight, lambda_weight, delta, tol_cg, tol_iter, max_cg, max_iter)   # restoration result by ADMM iteration

# evaluation
damge_psnr = mysubroutines.psnr(image_damage, image_true)
damge_ssim = ssim(image_damage, image_true, win_size=15)

end_psnr = mysubroutines.psnr(result, image_true)
end_ssim = ssim(result, image_true, win_size=15)

promotion = end_psnr-damge_psnr

print('PSNR: damge=%f, restoration=%f' %(damge_psnr,end_psnr))
print('SSIM: damge=%f, restoration=%f' %(damge_ssim,end_ssim))
print('Promotion = %f' % promotion)

# Plot
h = plt.figure()
fig_true = h.add_subplot(131)
fig_true.imshow(image_true, cmap="gray")
fig_true.set_title('true')
fig_true.xaxis.set_ticks([])
fig_true.yaxis.set_ticks([])

fig_damage = h.add_subplot(132)
fig_damage.imshow(image_damage, cmap="gray")
fig_damage.set_title('damage')
fig_damage.xaxis.set_ticks([])
fig_damage.yaxis.set_ticks([])

fig_restoration = h.add_subplot(133)
fig_restoration.imshow(result, cmap="gray")
fig_restoration.set_title('restoration')
fig_restoration.xaxis.set_ticks([])
fig_restoration.yaxis.set_ticks([])

h.savefig(output_image)
# plt.show()   # Show the plot locally