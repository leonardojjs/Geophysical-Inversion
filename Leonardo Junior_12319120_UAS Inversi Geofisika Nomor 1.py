#Nama: Leonardo Junior
#NIM: 12319120

import numpy as np
from sympy import symbols, diff, sqrt
import matplotlib.pyplot as plt
import scipy
from scipy import constants

#Setting Parameter
jml_iter = 10
titik_st = np.array([-500,-425,-350,-275,-200,-125,-50,25,100,175,250,325,400,475])
g_obs = np.array([0.0444,0.0619,0.0895,0.1346,0.2133,0.3553,0.6174,1.0648,1.5944,1.7061,1.2945,0.7443,0.4257,0.2514])
G = scipy.constants.gravitational_constant

#Setting Jac Matrix with Parameter
xr, R_bola, G0, x0, h = symbols('xr R_bola G x0 h')
g = (10**5)*(4/3)*np.pi*G0*2600*h*(R_bola**3)*((xr-x0)**2 + h**2)**(-1.5)
f1 = diff(g,x0)
f2 = diff(g,h)
f3 = diff(g,R_bola)

#Setting Parameter Model
model = np.zeros(3)
model[0] = 200; model[1]=150; model[2]=125
g_cal = np.zeros(len(titik_st))
dt = np.zeros(len(titik_st))
ff = np.zeros((len(titik_st), 3))
rms = np.zeros(jml_iter)

#Non Linear Inversion
for iter in range(jml_iter):
    for ii in range(len(titik_st)):
        g_cal[ii] = g.subs([(G0, G), (xr, titik_st[ii]), (x0, model[0]), (h, model[1]), (R_bola, model[2])])
        ff[ii,0] = f1.subs([(G0, G), (xr, titik_st[ii]), (x0, model[0]), (h, model[1]), (R_bola, model[2])])
        ff[ii,1] = f2.subs([(G0, G), (xr, titik_st[ii]), (x0, model[0]), (h, model[1]), (R_bola, model[2])])
        ff[ii,2] = f3.subs([(G0, G), (xr, titik_st[ii]), (x0, model[0]), (h, model[1]), (R_bola, model[2])])
    dt = g_obs-g_cal
    rms[iter] = np.sqrt(np.mean(dt ** 2))
    ft = np.transpose(ff)
    ftfinv = np.linalg.inv(np.dot(ft, ff))
    dm = np.dot(np.dot(ftfinv, ft), dt)
    model += dm
    print("Koordinat sumber pada iterasi", iter, model)
print('nilai rms pada tiap iterasi', rms)

#Plotting Anomaly Response and Calculation
plt.figure('Hasil Permodelan')
plt.scatter(titik_st, g_obs, color='blue')
plt.plot(titik_st, g_cal, color='red')
plt.legend(["Observation data", "Inversion modelling"])
plt.title('Gravity Anomaly Response')
plt.grid()
plt.show()

