import numpy as np
from sympy import symbols, diff, sqrt
import matplotlib.pyplot as plt

# Parameter data
jml_st = 10
jml_iter = 15
st_x_r = np.array([0, 0, 100, 100, 50, 80, 20, 79, 90, 5])
st_y_r = np.array([0, 100, 0, 100, 30, 30, 50, 24, 40, 98])
koordinat_x_sumber = 50
koordinat_y_sumber = 50

# Buat data sintetik untuk validasi model
v_untuk_st = 4
syntetic_time = np.zeros(jml_st)
for i in range(jml_st):
    syntetic_time[i] = sqrt((st_x_r[i] - koordinat_x_sumber) ** 2 + (st_y_r[i] - koordinat_y_sumber) ** 2) / v_untuk_st
print(syntetic_time)

# Buat Travel Time dengan noise gaussian
travel_time = np.array([17.67766953, 17.67766953, 17.67766953, 17.67766953, 5.0, 9.01387819, 7.5, 9.73717105, 10.30776406,
                        16.44878415])

# Definisikan mean dan deviasi standar untuk gaussian noise
mean = 0
std_dev = 0.1  # Sesuaikan deviasi standar sesuai kebutuhan

# Tambahkan gaussian noise ke travel time
travel_time_with_noise = travel_time + np.random.normal(mean, std_dev, travel_time.shape)

# Sampling random untuk model
model = np.random.randint(0, 100, size=(10000, 2))
model[1, 0] = 51
model[1, 1] = 49.5
print(model[0])

# Setting matriks kernel
vv = 4
xr, yr, xs, ys, v0 = symbols('xr yr xs ys vs')
dist = (xr - xs) ** 2 + (yr - ys) ** 2
ti = sqrt(dist) / v0
f1 = diff(ti, xs)
f2 = diff(ti, ys)
e = np.zeros(jml_st)
mitsfit = np.zeros(len(model))

# Inversi non-linear secara iteratif untuk tiap model random
for i in range(len(model)):
    ts = np.zeros(jml_st)
    ff = np.zeros((jml_st, 2))
    dt = np.zeros(jml_st)
    for iter in range(jml_iter):
        for ii in range(jml_st):
            ts[ii] = ti.subs([(v0, vv), (xr, st_x_r[ii]), (yr, st_y_r[ii]), (xs, model[i, 0]), (ys, model[i, 1])])
            ff[ii, 0] = f1.subs([(v0, vv), (xr, st_x_r[ii]), (yr, st_y_r[ii]), (xs, model[i, 0]), (ys, model[i, 1])])
            ff[ii, 1] = f2.subs([(v0, vv), (xr, st_x_r[ii]), (yr, st_y_r[ii]), (xs, model[i, 0]), (ys, model[i, 1])])
        dt = travel_time_with_noise - ts
        ft = np.transpose(ff)
        ftfinv = np.linalg.inv(np.dot(ft, ff))
        dm = np.dot(np.dot(ftfinv, ft), dt)
    model[i, 0] += dm[0]
    model[i, 1] += dm[1]
    e = (dt) ** 2
    mitsfit[i] = np.sum(e)

# Plot dan buat kontur untuk misfit
plt.scatter(st_x_r, st_y_r, color='blue')
plt.tricontourf(model[:, 0], model[:, 1], mitsfit)
plt.scatter(koordinat_x_sumber, koordinat_y_sumber, color='red')
plt.colorbar(label="mitsfit value", orientation="horizontal")
plt.contour
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('Peta Kontur Misfit dengan Gaussian Noise pada Travel Time')
plt.ylabel('Koordinat x')
plt.xlabel('Koordinat y')
plt.show()
