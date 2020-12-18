import csv
import glob
import matplotlib.pyplot as plt
import numpy as np

log_files = glob.glob('./data/*.csv')

# (12/15) preprocess for coroutine batch
samples=[]

for log_file in log_files:
    print(log_file)
    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        i=0
        for line in reader:
            if i > 1:
                samples.append(float(line[3]))
                if abs(samples[-1]) > 0.1:
                    samples.append(float(line[3])*(-1.0))
            i+=1
print(len(samples))
plt.hist(samples)

x=np.array([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]])

print(x[:,0])
plt.show()
