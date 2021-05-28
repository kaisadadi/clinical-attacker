import matplotlib.pyplot as plt
import numpy

#plt.rcParams['savefig.dpi'] = 300 #图片像素
#plt.rcParams['figure.dpi'] = 300 #分辨率
plt.rcParams['font.style'] = 'normal'

origin = [18.3, 42.7, 24.7, 9.2, 5.1]
new = [8, 22, 31, 21, 18]

plt.scatter([1, 2, 3, 4, 5], origin, marker='o', color='black')
plt.scatter([1, 2, 3, 4, 5], new, marker='o', color='black')
plt.plot([1, 2, 3, 4, 5], origin, color='red', label='Experiment #1', linestyle='-')
plt.plot([1, 2, 3, 4, 5], new, color='blue', label='Experiment #2', linestyle='-.')
plt.xticks([1, 2, 3, 4, 5], fontsize=12)
plt.yticks([5, 10, 15, 20, 25, 30, 35, 40, 45], fontsize=12)
plt.xlabel("Score", fontsize = 15)
plt.ylabel("Ratio(%)", fontsize=15)
plt.legend(fontsize=15)

plt.savefig("../fig.jpg", dpi=500)

plt.show()

