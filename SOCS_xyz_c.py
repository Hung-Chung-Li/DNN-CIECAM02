import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import math
import colour

 
dataFile_1 = 'C:/Users/hj123/Desktop/ACA_2019/Data.mat'
dataFile_2 = 'C:/Users/hj123/Desktop/ACA_2019/SOCS.mat'
data_1 = scio.loadmat(dataFile_1)
data_2 = scio.loadmat(dataFile_2)

Light = data_1['Light_source']
cmf = data_1['cmf']
SOCS = data_2['SOCS']

str = input("Light source No.?：");
no = int(str) - 1

k = 100/np.sum((Light[:,no]/np.max(Light[:,no]))*cmf[:,1])

Xw = np.sum((Light[:,no]/np.max(Light[:,no]))*cmf[:,0])*k
Yw = np.sum((Light[:,no]/np.max(Light[:,no]))*cmf[:,1])*k
Zw = np.sum((Light[:,no]/np.max(Light[:,no]))*cmf[:,2])*k

XYZ_d = []
for i in range(SOCS.shape[1]):
    X = np.sum((Light[:,no]/np.max(Light[:,no]))*cmf[:,0]*(SOCS[:,i]/np.max(SOCS[:,i])))*k
    Y = np.sum((Light[:,no]/np.max(Light[:,no]))*cmf[:,1]*(SOCS[:,i]/np.max(SOCS[:,i])))*k
    Z = np.sum((Light[:,no]/np.max(Light[:,no]))*cmf[:,2]*(SOCS[:,i]/np.max(SOCS[:,i])))*k
    XYZ_d.append([X,Y,Z])

XYZ_w = np.array([Xw,Yw,Zw])    
XYZ_d_ = np.array(XYZ_d)

N_A = Light[:,0]/np.max(Light[:,0])*198.261

t1 = np.arange(400, 710, 10)
plt.plot(t1,N_A, color='red', marker="", linestyle='dashdot')
plt.plot(t1,Light[:,1], marker="", linestyle='dashdot')
plt.plot(t1,Light[:,2], marker="", linestyle='dashdot')
plt.xlabel('Wavelength')
plt.ylabel('Relative power')
plt.legend(['A', 'D50', 'D65'], loc='upper left')
plt.show()

#/ Generate input data with CIECAM02 model 
La = Yw/5
Yb = 20

Jab_ = []
viewing_condition = ['Average', 'Dim', 'Dark']
for j in viewing_condition:
    surround = colour.CIECAM02_VIEWING_CONDITIONS[j]
    ans = colour.XYZ_to_CIECAM02(XYZ_d_, XYZ_w.T, La, Yb, surround)
    J = ans[0].T
    a = ans[1].T*np.cos(ans[2].T/(180/math.pi))
    b = ans[1].T*np.sin(ans[2].T/(180/math.pi))
    Jab = np.c_[J,a,b]
    Jab_.append(Jab)
    
Jab_o = np.concatenate((Jab_[0],Jab_[1],Jab_[2]),axis=0)