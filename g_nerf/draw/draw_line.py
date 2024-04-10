import numpy as np
import matplotlib.pyplot as plt

color_nomarker= ['#FF001E', '#0600F9', '#9400C3', '#5c3c92','#e8d21d', '#d72631', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markerlist=['v','<','^','s','D','<']
plt.switch_backend('agg')
# plt.style.use('seaborn-whitegrid')
font1 = {'family' : 'sans-serif',
'weight' : 'normal',
'size'   : 15,
}
font2 = {'family' : 'sans-serif',
'weight' : 'normal',
'size'   : 15,
}
f = plt.figure()
plt.grid(linestyle='--')
plt.rcParams["font.family"] = "Times New Roman"
#super resolution
# y2=[0.60893,0.62660,0.63606,0.64234,0.64684,0.65022, 0.65135, 0.65303, 0.65658, 0.65576]
# y1=[0.60736,0.62525,0.63395,0.64003,0.64524,0.64812, 0.64969, 0.65347, 0.65569, 0.65696]
# y0=[0.60915,0.62423,0.63260,0.63742,0.64347,0.64730, 0.64830, 0.65268, 0.65461, 0.65558]
#laten code dimension
y2=[0.608943164348602,0.625799059867858,0.63606,0.64234,0.64684,0.65022, 0.65135, 0.65303, 0.65658, 0.65576]
y1=[0.616487026214599,0.629931449890136,0.63395,0.64003,0.64524,0.64812, 0.64969, 0.65347, 0.65569, 0.65696]
y0=[0.616087079048156,0.627610981464386,0.63260,0.63742,0.64347,0.64730, 0.64830, 0.65268, 0.65461, 0.65558]
x = np.arange(10)
x_general=[201,403,604,806,1008,1209,1411,1612,1814,2016]
interval = 1
x_general = x_general[::interval]
y0 = y0[::interval]
y1 = y1[::interval]
y2 = y2[::interval]
# x_general = x_general[2:]
# y0 = y0[2:]
# y1 = y1[2:]
# y2 = y2[2:]
# plt.title(' SR ', y=-0.19,fontsize=21)
# plt.suptitle('test title', fontsize=20)
plt.plot(x_general, y0, color=color_nomarker[0], marker=markerlist[0], label='3 layers', linewidth=2, markersize=10, fillstyle='none')
plt.plot(x_general, y1, color=color_nomarker[1], marker=markerlist[1], label='5 layers', linewidth=2, markersize=10, fillstyle='none')
plt.plot(x_general, y2, color=color_nomarker[2], marker=markerlist[2], label='7 layers', linewidth=2, markersize=10, fillstyle='none')

plt.xlabel('Training step(k images)',font2)
plt.ylabel('SSIM',font2)
plt.legend(loc='lower right',fontsize=15,frameon=True)
plt.savefig('test.jpg')
plt.savefig('test.pdf')