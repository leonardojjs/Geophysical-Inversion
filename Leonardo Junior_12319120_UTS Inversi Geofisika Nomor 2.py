import numpy as np
import matplotlib.pyplot as plt

# Parameter koordinat source, receiver, waktu observasi,dan model kecepatan awal
sx = np.array([300, 100, 0])
sy = np.array([0, 0, 0])
rx = np.array([400, 400, 300, 100])
ry = np.array([200, 400, 400, 400])
tobs = np.array([0.2111, 0.2364, 0.2235, 0.1999, 0.2368, 0.2836, 0.2069])
v_awal = np.array([1850, 1500, 1900, 1900])

#Mengatur matriks waktu kalkulasi, panjang lintasan, dan selisih waktu tempuh observasi dengan kalkulasi
tcal = np.zeros(len(tobs))
l = np.zeros(len(tobs))
dt = np.zeros(len(tobs))

#Menghitung panjang lintasan untuk 7 sinar seismik
l[0] = ((rx[2] - sx[0]) ** 2 + (ry[2] - sy[0]) ** 2)**(0.5)
l[1] = ((rx[3] - sx[0]) ** 2 + (ry[3] - sy[0]) ** 2)**(0.5)
l[2] = ((rx[2] - sx[1]) ** 2 + (ry[2] - sy[1]) ** 2)**(0.5)
l[3] = ((rx[3] - sx[1]) ** 2 + (ry[3] - sy[1]) ** 2)**(0.5)
l[4] = ((rx[0] - sx[2]) ** 2 + (ry[0] - sy[2]) ** 2)**(0.5)
l[5] = ((rx[1] - sx[2]) ** 2 + (ry[1] - sy[2]) ** 2)**(0.5)
l[6] = ((rx[3] - sx[2]) ** 2 + (ry[3] - sy[2]) ** 2)**(0.5)

#Menghitung waktu tempuh kalkulasi dengan mengasumsikan panjang lintasan yang melewati 2 kotak sama besar
tcal[0] = (l[0]/2)*((1/v_awal[1])+(1/v_awal[3]))
tcal[1] = (l[1]/2)*((1/v_awal[1])+(1/v_awal[2]))
tcal[2] = (l[2]/2)*((1/v_awal[0])+(1/v_awal[3]))
tcal[3] = (l[3]/2)*((1/v_awal[0])+(1/v_awal[2]))
tcal[4] = (l[4]/2)*((1/v_awal[0])+(1/v_awal[1]))
tcal[5] = (l[5]/2)*((1/v_awal[0])+(1/v_awal[3]))
tcal[6] = (l[6]/2)*((1/v_awal[0])+(1/v_awal[2]))

#Menghitung beda waktu tempuh kalkulasi dengan observasi sebagai input matriks d
for i in range(len(tobs)):
  dt[i] = tobs[i] - tcal[i]

#Setting matriks kernel G dengan input panjang lintasan
G = np.array([[0, l[0]/2, 0, l[0]/2],
              [0, l[1]/2, l[1]/2, 0],
              [l[2]/2, 0, 0, l[2]/2],
              [l[3]/2, 0, l[3]/2, 0],
              [l[4]/2, l[4]/2, 0, 0],
              [l[5]/2, 0, 0, l[5]/2],
              [l[6]/2, 0, l[6]/2, 0]])

#Persamaan transpose matriks G
GT = np.transpose(G)
epsilon = np.linspace(0.001, 0.1, num=100)
A = np.dot(GT, G)

#Setting parameter matriks model estimasi (mest) sebagai delta slowness, matriks waktu tempuh kalkulasi final, dan juga mitsfit
mest = np.zeros(len(epsilon))
id = np.identity(len(A))
v1 = np.zeros(len(epsilon))
v2 = np.zeros(len(epsilon))
v3 = np.zeros(len(epsilon))
v4 = np.zeros(len(epsilon))
tcal_final1 = np.zeros(len(epsilon))
tcal_final2 = np.zeros(len(epsilon))
tcal_final3 = np.zeros(len(epsilon))
tcal_final4 = np.zeros(len(epsilon))
tcal_final5 = np.zeros(len(epsilon))
tcal_final6 = np.zeros(len(epsilon))
tcal_final7 = np.zeros(len(epsilon))
misfit = np.zeros(len(epsilon))

#Menghitung solusi slowness untuk tiap blok dengan asumsi perubahan kecepatan yang kecil atau mendekati 0 dengan penambahan bobot
#Menghitung mitsfit paling kecil pada saat nilai redaman yang berkisar antara 0.001 hingga 0.1
for i in range(len(epsilon)):
  mest = np.dot((np.dot((np.linalg.inv(A + ((epsilon[i]**2)*id))), GT)), dt)
  v1[i] = v_awal[0] + (-mest[0]*v_awal[0]**2)
  v2[i] = v_awal[1] + (-mest[1]*v_awal[1]**2)
  v3[i] = v_awal[2] + (-mest[2]*v_awal[2]**2)
  v4[i] = v_awal[3] + (-mest[3]*v_awal[3]**2)
  tcal_final1[i] = (l[0]/2)*((1/v2[i])+(1/v4[i]))
  tcal_final2[i] = (l[1]/2)*((1/v2[i])+(1/v3[i]))
  tcal_final3[i] = (l[2]/2)*((1/v1[i])+(1/v4[i]))
  tcal_final4[i] = (l[3]/2)*((1/v1[i])+(1/v3[i]))
  tcal_final5[i] = (l[4]/2)*((1/v1[i])+(1/v2[i]))
  tcal_final6[i] = (l[5]/2)*((1/v1[i])+(1/v4[i]))
  tcal_final7[i] = (l[6]/2)*((1/v1[i])+(1/v3[i]))
  misfit[i] = ((tobs[0]-tcal_final1[i])**2 + (tobs[1]-tcal_final2[i])**2 + (tobs[2]-tcal_final3[i])**2 + (tobs[3]-tcal_final4[i])**2 + (tobs[4]-tcal_final5[i])**2 + (tobs[5]-tcal_final6[i])**2 + (tobs[6]-tcal_final7[i])**2 )/epsilon[i]**2

#Plot kurva epsilon terhadap mitsfit
plt.plot(misfit, epsilon)
plt.xlabel("mitsfit")
plt.ylabel("epsilon")
plt.show()

#Mengambil nilai mitsfit paling awal dan akhir
print("Mitsfit pertama saat epsilon 0.001 adalah", + misfit[0])
print("Mitsfit terakhir saat epsilon 0.1 adalah", +misfit[-1])
print("Dalam kasus ini apabilan melihat pada kurva plot ditarik kesimpulan bahwa saat nilai redaman membesar, maka nilai mitsfit akan mendekati 0")
print("Oleh sebab itu, saya memilih nilai redaman 0.1 untuk menghasilkan mitsfit paling kecil")
print("Maka model kecepatan akhir untuk v1, v2, v3, v4 secara berurutan adalah", + v1[-1], v2[-1], v3[-1], v4[-1])