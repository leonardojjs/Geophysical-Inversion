#Nama: Leonardo Junior
#NIM: 12319120

import numpy as np
from sympy import symbols, diff, sqrt
import matplotlib.pyplot as plt
import scipy
from scipy import constants
import random

#Setting Parameter
jml_iter = 20
titik_st = np.linspace(-1000,1000, num=100)
G = scipy.constants.gravitational_constant
x0 = 500
h0 = 500

#Create Syntetic Data
g_sintetik = np.zeros(len(titik_st))
g_sintetik_noise = np.zeros(len(titik_st))

for i in range(len(titik_st)):
    g_sintetik[i] = (10**5)*(4/3) * np.pi * G * 2800 * h0 * (100 ** 3) * ((titik_st[i] - x0) ** 2 + h0 ** 2) ** (-1.5)
#Give Random Noise to Syntetic Data
    g_sintetik_noise[i] = g_sintetik[i] + random.gauss(0, 0.01)
print(g_sintetik)

#Setting Jac Matrix
xr, R_bola, G0, x0, h = symbols('xr R_bola G x0 h')
g = (10**5)*(4/3)*np.pi*G0*2800*h*(100**3)*((xr-x0)**2 + h**2)**(-1.5)
f1 = diff(g,x0)
f2 = diff(g,h)

#Setting Parameter Model
model = np.zeros(2)
model[0] = 470; model[1]=530
g_cal = np.zeros(len(titik_st))
dt = np.zeros(len(titik_st))
ff = np.zeros((len(titik_st),2))
rms = np.zeros(jml_iter)

#Non Linear Inversion
for iter in range(jml_iter):
    for ii in range(len(titik_st)):
        g_cal[ii] = g.subs([(G0, G), (xr, titik_st[ii]), (x0, model[0]), (h, model[1])])
        ff[ii,0] = f1.subs([(G0, G), (xr, titik_st[ii]), (x0, model[0]), (h, model[1])])
        ff[ii,1] = f2.subs([(G0, G), (xr, titik_st[ii]), (x0, model[0]), (h, model[1])])
    dt = g_sintetik_noise-g_cal
    rms[iter] = np.sqrt(np.mean(dt**2))
    ft = np.transpose(ff)
    ftfinv = np.linalg.inv(np.dot(ft,ff))
    dm = np.dot(np.dot(ftfinv,ft), dt)
    model += dm
    print("Koordinat sumber pada iterasi", iter, model)
print('nilai rms pada tiap iterasi', rms)

#Plotting Anomaly Response and Calculation
plt.figure('Result of Modelling')
plt.scatter(titik_st, g_sintetik, color='red')
plt.scatter(titik_st, g_sintetik_noise, color='blue')
plt.plot(titik_st, g_cal, color='green')
plt.legend(["Forward modelling" , "Observation data with noise", "Inversion modelling"])
plt.title('Gravity Anomaly Response')
plt.grid()
plt.xlim(-1000,1000)
plt.show()