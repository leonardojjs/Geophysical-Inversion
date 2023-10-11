# -*- coding: utf-8 -*-
"""Tugas 1 Inversi

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QZTxlGM_kMKosiNIz08LoekdeSgnzMne
"""

#Tugas 1
#Mata Kuliah: TG4141 Inversi Geofisika
#NIM        : 12319120
#Nama       : Leonardo Junior Johan Solihin

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import os
from matplotlib.image import NonUniformImage
from matplotlib.ticker import NullFormatter
from pylab import *
import scipy
from scipy import constants

G = scipy.constants.gravitational_constant
R = 100
rho = 2800
Xc = 500
Zc = 500
n = 64

#Fungsi Respon Anomali Graviti
def f(x):
  return G*(4*np.pi*(R**3)*rho*Zc)/(((x-Xc)**2 + Zc**2)**1.5)

Xi = np.linspace(0,1000,num=1000)
gz = f(Xi)

#Persamaan Lingkaran
t = np.linspace(0, 2*np.pi, n+1)
x = R*np.cos(t)+500
y = R*np.sin(t)-500

plt.figure(1, dpi=120)

#1
plt.subplot2grid((2,2), (0,0))
plt.plot(Xi,gz)
plt.title('Gravity Anomaly Response')
plt.xlabel('(m) ')
plt.ylabel('g (gal)')

#2
plt.subplot2grid((2,2), (1,0))
plt.plot(x,y)
plt.ylim(-1100,100)
plt.xlim(-100,1100)
plt.title('Subsurface Model')
plt.xlabel('(m)')
plt.ylabel('(m) ')


plt.gca().yaxis.set_minor_formatter(NullFormatter())
plt.subplots_adjust(top=20, bottom=19, left=10, right=11, hspace=0.5,
                    wspace=0.35)
plt.show()

plt.figure(dpi=120)
plt.subplot(2, 1, 1) # row 1, col 2 index 1
plt.plot(Xi, gz)
plt.title('Gravity Anomaly Response')
plt.xlabel('(m) ')
plt.ylabel('g (gal)')

plt.subplot(2, 1, 2) # index 2
plt.plot(x, y)
plt.ylim(-1100,100)
plt.xlim(-100,1100)
plt.title('Subsurface Model')
plt.xlabel('(m)')
plt.ylabel('(m) ')

plt.subplots_adjust(hspace=0.6,
                    wspace=0.35)

plt.show()