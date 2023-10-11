import numpy as np

#Parameter data
x1 = np.array([-2.2, -0.25, 0, 1, 1.25, 5.25])
x2 = np.array([-2.25, -1.2, 0, 1, 4.25, 5])
y = np.array([7.37376, 1.79669, 0.137058, -0.456891, 11.9378, 11.0596])

#Setting matriks kernel G
G = np.ones((6, 3))
G[:, 1] = x1
G[:, 2] = x2**2

#Transpose matriks G dan inversi matriks
GT = np.transpose(G)
Ginv = np.linalg.inv(np.dot(GT, G))

#a. Menghitung solusi model estimasi (m) untuk menentukan koesfisien a0, a1, a2
mest = np.dot(Ginv, np.dot(GT, y))
print("Maka koefisien a0 =", + mest[0])
print("Maka koefisien a1 =", + mest[1])
print("Maka koefisien a2 =", + mest[2])

#Y hasil solusi
y_solusi = mest[0] + mest[1]*x1 + mest[2]*x2**2

#b. Menghitung error dan mitsfit
e = np.zeros(6)
for i in range(len(y)):
  e[i] = (y[i]-y_solusi[i])**2
mitsfit = np.sum(e)
print("Nilai Mitsfit dari solusi inversi di atas adalah", + mitsfit)

#c. Laju_Perubahan (turunan fungsi penjumlahan)
Laju = mest[1] + 2*mest[2]*0.5
print("laju perubahan pada saat posisi (-2,0.5) adalah", + Laju)

#d. nilai y pada koordinat (-1.5,-1.5), (-2,0.5), (3.5,4.0)
print("nilai y pada koordinat (-1.5,-1.5) adalah", mest[0] + mest[1]*(-1.5) + mest[2]*(-1.5)**2)
print("nilai y pada koordinat (-2,0.5) adalah", mest[0] + mest[1]*(-2) + mest[2]*(0.5)**2)
print("nilai y pada koordinat (3.5,4.0) adalah", mest[0] + mest[1]*(3.5) + mest[2]*(4)**2)
