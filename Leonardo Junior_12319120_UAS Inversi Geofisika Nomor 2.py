import numpy as np
import matplotlib.pyplot as plt

#Data Lubang Bor
z = [10, 35, 40, 60, 80]
T = [41.7, 47.7, 49.2, 56.2, 64]

#fungsi Misfit
def misfit(a0, a1, a2):
    eror2 = 0
    for k in range(len(z)):
        eror2 += (T[k] - (a0 + a1*z[k] + a2*(z[k]**2)))**2
    return eror2

#make grid
delta_a1 = 0.001
delta_a2 = 0.0001
delta_a0 = 0.4

#Mengisi Nilai Grid
ni = np.arange(30,50+delta_a0,delta_a0)
mi1 = np.arange(0.1,0.2+delta_a1,delta_a1)
mi2 = np.arange(0.0,0.003+delta_a2,delta_a2)
mv1, mv2, nv = np.meshgrid(mi1, mi2, ni)
tv = np.zeros((len(ni), len(mi1), len(mi2)))
for i in range(len(ni)):
    for j in range(len(mi1)):
        for k in range(len(mi2)):
            tv[i][j][k] = misfit(ni[i], mi1[j], mi2[k])

print("Misfit pada setiap koordinat : ",tv)
fig = plt.figure('Hasil Plot Misfit untuk tiap nilai a0, a1, a2')
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter3D(mv1, mv2, nv, c=tv, cmap='OrRd', alpha=0.9)
ax.set_xlabel('Gradient a1')
ax.set_ylabel('Gradient a2')
ax.set_zlabel('Suhu Permukaan a0')
fig.colorbar(img, ax=ax, label='Nilai Misfit', orientation='horizontal')
plt.title('Hasil Perhitungan untuk tiap solusi')
plt.show()