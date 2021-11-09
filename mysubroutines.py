import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# get the Gaussian kernel
def gaussian_kernel(kernel_size, sigma):
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    ker = kx * ky.T
    return ker

# rename cv2.filter2D, choose the borderType
def convolution(image, kernel):
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_WRAP)

# transpose of convolution
def convolution_T(image, kernel):
    return cv2.filter2D(image, -1, cv2.flip(kernel,-1), borderType=cv2.BORDER_WRAP)

# get finite difference operator x
def Wx(image):
    kernel = np.array([[0,0,0],[0,-1,0],[0,1,0]])
    Wx = cv2.filter2D(image, -1, kernel=kernel)
    return  Wx

# get transpose of finite difference operator x
def Wx_T(image):
    kernel = np.array([[0,1,0],[0,-1,0],[0,0,0]])
    WxT = cv2.filter2D(image, -1, kernel=kernel)
    return WxT

# get finite difference operator y
def Wy(image):
    kernel = np.array([[0,0,0],[0,-1,1],[0,0,0]])
    Wy = cv2.filter2D(image, -1, kernel=kernel)
    return  Wy

# get transpose of finite difference operator y
def Wy_T(image):
    kernel = np.array([[0,0,0],[1,-1,0],[0,0,0]])
    WyT = cv2.filter2D(image, -1, kernel=kernel)
    return WyT

# operator A of equation Ax=b
def operator(image, kernel, mu):
    return convolution_T(convolution(image, kernel), kernel)+mu*Wx_T(Wx(image))+mu*Wy_T(Wy(image))

# Conjugate Gradient iteration to solve Ax=b
def conj_grad(kernel, f, mu, dx, bx, dy, by, tol_cg, max_cg):
    x = f
    r = convolution_T(f, kernel)+mu*Wx_T(dx-bx)+mu*Wy_T(dy-by)-operator(x,kernel,mu)
    rsold = np.sum(r*r)
    p = r
    k = 0
    while k<max_cg:
        Ap = operator(p,kernel,mu)
        alpha = np.sum(r*r)/np.sum(p*Ap)
        x = x + alpha*p
        r = r - alpha*Ap
        rsnew = np.sum(r*r)
        if np.sqrt(rsnew)<tol_cg:   # tolerance
            break
        b = rsnew/rsold
        p = r+b*p
        rsold = rsnew
        k += 1
    return x

# Calculate the Peak Signal to Noise Ratio of the image_new, compared with the image_true
def psnr(image_new, image_true):
    d = len(image_true)
    mse = np.sum((image_new-image_true)**2)/d**2
    psnr_u = 10 * np.log10(np.max(image_new) ** 2 / mse)
    return psnr_u

# Split Bregman (ADMM)
def ADMM(image_true, kernel, f, mu, lam, delta, tol_cg, tol_iter, max_cg, max_iter):
    d = len(f)
    # initial
    u = f
    dx = np.zeros((d, d))
    bx = np.zeros((d, d))
    dy = np.zeros((d, d))
    by = np.zeros((d, d))
    k = 0   # iteration times
    while k<max_iter:
        u = conj_grad(kernel, u, mu, dx, bx, dy, by, tol_cg, max_cg)
        # shrink
        sx = Wx(u) + bx
        sy = Wy(u) + by
        s = np.sqrt(np.sum(sx ** 2) + np.sum(sy ** 2))
        dx = max(s - lam / mu, 0) / s * sx
        dy = max(s - lam / mu, 0) / s * sy
        bx += delta*(Wx(u)-dx)
        by += delta*(Wy(u)-dy)
        k += 1
        # evaluation
        psnr_u = psnr(u, image_true)
        ssim_u = ssim(u, image_true, win_size=15)
        print('iter=%s, PSNR=%f, SSIM=%f' % (k, psnr_u, ssim_u))
        error = np.sum((Wx(u)-dx)**2+(Wy(u)-dy)**2)/np.sum(f**2)
        if error<tol_iter:   # tolerance
            print('ADMM iter times=', k)
            print('error=',error)
            break
    return u