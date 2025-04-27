import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.fft import fftshift, fft2, ifft2, ifftshift
from torchvision import transforms

class Lens(nn.Module):
    def __init__(self, whole_dim, pixel_size, focal_length, wave_lambda):
        super(Lens, self).__init__()
        # basic parameters
        temp = np.arange((-np.ceil((whole_dim - 1) / 2)),
                        np.floor((whole_dim - 1) / 2)+0.5)
        x = temp * pixel_size
        xx, yy = np.meshgrid(x, x)
        lens_function = np.exp(
            -1j * math.pi / wave_lambda / focal_length * (xx ** 2 + yy ** 2))
        self.lens_function = torch.tensor(
            lens_function, dtype=torch.complex64).cuda()

    def forward(self, input_field):
        out = torch.mul(input_field, self.lens_function)
        return out

class ASM_prop_layer(nn.Module):
    def __init__(self, whole_dim, pixel_size, focal_length, wave_lambda, refidx = 1): #32 400000  # 8 200000 200
        super().__init__()
        self.input_size = [whole_dim, whole_dim]           # input_size * input_size neurons in one layer
        # self.z = focal_length * 1e-6  # distance bewteen two layers
        # self.pixelsize = pixel_size * 1e-6       # pixel size
        # self.lamda = wave_lambda * 1e-9               # wavelength
        self.z = focal_length * 1 # distance bewteen two layers
        self.pixelsize = pixel_size * 1      # pixel size
        self.lamda = wave_lambda * 1              # wavelength

        NFv,NFh = self.input_size[0], self.input_size[1]
        Fs = 1 / self.pixelsize
        Fh = Fs / NFh * np.arange((-np.ceil((NFh - 1) / 2)), np.floor((NFh - 1) / 2)+0.5)
        Fv = Fs / NFv * np.arange((-np.ceil((NFv - 1) / 2)), np.floor((NFv - 1) / 2)+0.5)
        [Fhh, Fvv] = np.meshgrid(Fh, Fv)
        
        np_H = self.PropGeneral(Fhh, Fvv, self.lamda, refidx, self.z)
        np_freqmask = self.BandLimitTransferFunction(self.pixelsize, self.z, self.lamda, Fvv, Fhh)

        self.H = torch.tensor(np_H, dtype=torch.complex64).cuda()
        self.freqmask = torch.tensor(np_freqmask, dtype=torch.complex64).cuda()
        
        # self.H = torch.tensor(np_H, dtype=torch.complex64).to(device)
        # self.freqmask = torch.tensor(np_freqmask, dtype=torch.complex64).to(device)
    def PropGeneral(self, Fhh, Fvv, lamda, refidx, z):
        DiffLimMat = np.ones(Fhh.shape)
        lamdaeff = lamda / refidx
        DiffLimMat[(Fhh ** 2.0 + Fvv ** 2.0) >= (1.0 / lamdaeff ** 2.0)] = 0.0

        temp1 = 2.0 * math.pi * z / lamdaeff
        temp3 = (lamdaeff * Fvv) ** 2.0
        temp4 = (lamdaeff * Fhh) ** 2.0
        temp2 = np.complex128(1.0 - temp3 - temp4) ** 0.5
        H = np.exp(1j * temp1*temp2)
        H[np.logical_not(DiffLimMat)] = 0
        return H

    def BandLimitTransferFunction(self, pixelsize, z, lamda, Fvv, Fhh):
        hSize, vSize = Fvv.shape
        dU = (hSize * pixelsize) ** -1.0
        dV = (vSize * pixelsize) ** -1.0
        Ulimit = ((2.0 * dU * z) ** 2.0 + 1.0) ** -0.5 / lamda
        Vlimit = ((2.0 * dV * z) ** 2.0 + 1.0) ** -0.5 / lamda
        freqmask = ((Fvv ** 2.0 / (Ulimit ** 2.0) + Fhh ** 2.0 * (lamda ** 2.0)) <= 1.0) & ((Fvv ** 2.0 * (lamda ** 2.0) + Fhh ** 2.0 / (Vlimit ** 2.0)) <= 1.0)
        return freqmask

    def forward(self, waves, use_freqmask=True):
        spectrum = torch.fft.fftshift(torch.fft.fft2(waves))
        spectrum_z = torch.mul(spectrum, self.H)
        if use_freqmask is True:
            spectrum_z = torch.mul(spectrum_z, self.freqmask)
        wave_z = torch.fft.ifft2(torch.fft.ifftshift(spectrum_z))
        return wave_z

class AngSpecProp(nn.Module):  # based on Matlab
    def __init__(self, whole_dim, pixel_size, focal_length, wave_lambda, n=1.0):
        super(AngSpecProp, self).__init__()
        wave_lambda_m = wave_lambda / n
        k = 2*math.pi/wave_lambda_m  # optical wavevector
        f = np.fft.fftshift(np.fft.fftfreq(whole_dim, d=pixel_size))
        fxx, fyy = np.meshgrid(f, f)
        angspec = np.maximum(0, k**2 - (2 * math.pi * fxx)**2 - (2 * math.pi * fyy)**2)
        self.Q2 = np.exp(1j * focal_length * np.sqrt(angspec))
        self.Q2 = torch.tensor(self.Q2).cuda()
        self.pixel_size = pixel_size

    def forward(self, input_field):
        A0 = fftshift(fft2(input_field))
        Az = A0 * self.Q2
        Uz = ifft2(ifftshift(Az))
        return Uz

class FresnelProp(nn.Module):  # based on Matlab
    def __init__(self, whole_dim, pixel_size, focal_length, wave_lambda):
        super(FresnelProp, self).__init__()
        k = 2*math.pi/wave_lambda  # optical wavevector
        df1 = 1 / (whole_dim*pixel_size)
        f = np.arange((-np.ceil((whole_dim - 1) / 2)),
                      np.floor((whole_dim - 1) / 2)+0.5) * df1
        fxx, fyy = np.meshgrid(f, f)
        fsq = fxx ** 2 + fyy ** 2

        self.Q2 = torch.tensor(
            np.exp(-1j*(math.pi**2)*2*focal_length/k*fsq), dtype=torch.complex64).cuda()
        self.pixel_size = pixel_size
        self.df1 = df1

    def ft2(self, g, delta):
        return fftshift(fft2(ifftshift(g))) * (delta ** 2)

    def ift2(self, G, delta_f):
        N = G.shape[1]
        return ifftshift(ifft2(fftshift(G))) * ((N * delta_f)**2)

    def forward(self, input_field):
        # compute the propagated field
        Uout = self.ift2(self.Q2 * self.ft2(input_field,
                        self.pixel_size), self.df1)
        return Uout