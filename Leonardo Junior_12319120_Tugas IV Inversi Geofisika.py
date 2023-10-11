import numpy as np
from matplotlib import pyplot as plt, patches
import scipy
from scipy import constants

fig, (ax2, ax1) = plt.subplots(2,1)
x = np.arange(-100,100, 20, dtype=int)
y = np.arange(-200,0, 20, dtype=int)
for i in range(len(x)):
    for j in range(len(x)):
        ax1.add_patch(plt.Circle((x[i], y[j]), 10, color='r'))

ax1.add_patch(plt.Circle((-80, -40), 10, color='yellow'))
ax1.add_patch(plt.Circle((-80, -60), 10, color='yellow'))
ax1.add_patch(plt.Circle((-40, -20), 10, color='yellow'))
ax1.add_patch(plt.Circle((-20, -40), 10, color='yellow'))
ax1.add_patch(plt.Circle((-20, -60), 10, color='yellow'))
ax1.add_patch(plt.Circle((-40, -80), 10, color='yellow'))
ax1.add_patch(plt.Circle((20, -200), 10, color='yellow'))
ax1.add_patch(plt.Circle((40, -180), 10, color='yellow'))
ax1.add_patch(plt.Circle((60, -180), 10, color='yellow'))
ax1.add_patch(plt.Circle((60, -140), 10, color='yellow'))
ax1.add_patch(plt.Circle((60, -200), 10, color='yellow'))
ax1.add_patch(plt.Circle((20, -160), 10, color='yellow'))
ax1.add_patch(plt.Circle((80, -160), 10, color='yellow'))

G = scipy.constants.gravitational_constant
R = 10
rho = 3200
Xc = [-80,-80,-40,-20,-20,-40, 20, 40, 60, 60, 60, 20, 80]
Zc = [-40,-60,-20,-40,-60,-80, -200, -180, -180, -140, -200, -160, -160]

#Fungsi Respon Anomali Graviti
xi = np.linspace(-100,100,num=1000)
gz = []
graviti = np.zeros(len(Xc))
for i in range(len(xi)):
    for j in range(len(Xc)):
        graviti[j] = abs(G*(4*np.pi*(R**3)*rho*Zc[j])/(((xi[i]-Xc[j])**2 + Zc[j]**2)**1.5))
    gz.append(sum(graviti)*1000)
print(gz)

ax1.set_aspect("equal", adjustable="datalim")
ax1.set_box_aspect(0.5)
ax1.autoscale()
ax1.set_xlabel('Koordinat X (m)')
ax1.set_ylabel('Kedalaman (m)')
ax1.set_title('Model Geologi Bawah Permukaan Provinsi Guangzhou China')

ax2.plot(xi,gz)
ax2.set_title('Fungsi Anomali Respon Graviti')
ax2.set_xlabel('Koordinat X (m)')
ax2.set_ylabel('Graviti (mgal)')
plt.subplots_adjust(left=0.2,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.5)
plt.show()