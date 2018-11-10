from scipy import spatial
import matplotlib.pyplot as plt
import numpy as np

dataSetI = np.asarray([50, 1, 2, 4])
dataSetII = np.asarray([2, 54, 13, 15])
result = 1 - spatial.distance.cosine(dataSetI, dataSetII)

print(dataSetII - dataSetI)
print(result)

plt.figure('1')
plt.plot(dataSetI)
plt.plot(dataSetII)

plt.figure('2')
plt.plot((dataSetII-dataSetI))

plt.show()