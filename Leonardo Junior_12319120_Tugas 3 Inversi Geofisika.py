import numpy as np
from sympy import symbols, diff, sqrt
import matplotlib.pyplot as plt

jml_st = 4
jml_iter = 10
st = np.zeros((jml_st,2))
st[0,0]=20; st[0,1]=15
st[1,0]=50; st[1,1]=25
st[2,0]=45; st[2,1]=50
st[3,0]=20; st[3,1]=45

travel_time = np.zeros(jml_st)
travel_time[0] =11.31 ; travel_time[1] = 10.75; travel_time[2] = 5.59
travel_time[3] = 3.95

vv = 4
xr,yr, xs, ys, v0 = symbols('xr yr xs ys vs')
dist = (xr-xs)**2 + (yr-ys)**2
ti = sqrt(dist)/v0

f1 = diff(ti,xs)
f2 = diff(ti,ys)

model = np.zeros(2)
model[0] = 25; model[1]=60
ts = np.zeros(jml_st)
dt = np.zeros(jml_st)
ff = np.zeros((jml_st,2))
rms = np.zeros(jml_iter)

for iter in range(10):
    for ii in range(jml_st):
        ts[ii] = ti.subs([(v0, vv), (xr, st[ii,0]),(yr, st[ii,1]), (xs, model[0]), (ys, model[1])])
        ff[ii,0] = f1.subs([(v0, vv), (xr, st[ii,0]),(yr, st[ii,1]), (xs, model[0]), (ys, model[1])])
        ff[ii,1] = f2.subs([(v0, vv), (xr, st[ii,0]),(yr, st[ii,1]), (xs, model[0]), (ys, model[1])])
        print(ts)
    dt = travel_time-ts
    rms[iter] = np.sqrt(np.mean(dt**2))
    ft = np.transpose(ff)
    ftfinv = np.linalg.inv(np.dot(ft,ff))
    dm = np.dot(np.dot(ftfinv,ft), dt)
    print(dm)
    model += dm
    print("Koordinat sumber pada iterasi", iter, model)
print('nilai rms pada tiap iterasi', rms)
