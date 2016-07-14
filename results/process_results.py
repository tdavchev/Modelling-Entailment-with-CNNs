from sys import argv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



script, filename, which = argv

target = open(filename, 'r')

which = int(which)
idx = -1
# skip the first row
first = 1
suma = 0
count = 0
for line in target:
    line = line.split()
    if line[0] == "epoch:":
        count += 1
        if which == 1:
            suma += float(line[8])
        elif which == 2:
            suma+=float(line[12])
        elif which == 3:
            suma+=float(line[4])
print suma/float(count)
import os 
dropout = []
dropout2 = []
learn_decay = []
learn_decay2 = []
accuracy = []
accuracy2 = []
count = 0
cur_count = 0
for fn in os.listdir('.'):
     if os.path.isfile(fn) and "job4" in fn:
        line_no = 0
        job = open(fn, 'r')
        marked = False
        count += 1
        shouldI = True

        add = False
        for line in job:
            if line_no == 1:
                if "200" in line:
                    shouldI = False
            if shouldI:
                if line_no == 9:
                    # print line
                    line = line.split()
                    for idx in xrange(0,len(line)):
                        if "add" in line[idx]:
                            add = True
                        if "dropout" in line[idx]:
                            digit = line[idx+1]
                            digit = digit.translate(None, '[],)')
                            # print digit
                            digit = digit.strip("]")
                            # print 
                            dropout.append(float(digit[:4]))

                            dropout2.append(float(digit[:4]))
                            # print "tuk sam"
                        elif "learn_decay" in line[idx]:
                            digit = line[idx+1]
                            digit = digit.translate(None, '[],)')
                            learn_decay.append(float(digit)) 
                            learn_decay2.append(float(digit))    
                    # print line
                elif len(line.split()) == 1:
                    # print line_no
                    if "epoch" in prev_line and "aha" not in line:
                        marked = True
                        cur_count += 1
                        if  add:
                            accuracy2.append(float(line))
                        else:
                            accuracy.append(float(line))

            line_no += 1
            prev_line = line

        if len(dropout) != len(accuracy):
            dropout = dropout[:-1]
            learn_decay = learn_decay[:-1]
        if len(dropout2) != len(accuracy2):
            dropout2 = dropout2[:-1]
            learn_decay2 = learn_decay2[:-1]

X = np.asarray(dropout)
Y = np.asarray(learn_decay)
Z = np.asarray(accuracy)

X2 = np.asarray(dropout2)
Y2 = np.asarray(learn_decay2)
Z2 = np.asarray(accuracy2)

# import matplotlib.pyplot as plt
# plt.plot(dropout,accuracy,'ro')
# plt.xlabel('dropout')
# plt.ylabel('accuracy')
# plt.title('Plotting dropout against accuracy')
# # plt.axis([0.6, 1, 0.6, 0.8])
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X, Y, Z, c='r', marker='o')
# ax.scatter(X2, Y2, Z2, c='b', marker='o')

# ax.set_xlabel('dropout')
# ax.set_ylabel('learn decay')
# ax.set_zlabel('percent accuracy')
# ax.set_title('Plotting dropout and learn decay against accuracy')

# plt.show()

results_no = [1,2,3,4,5,6]#,7]
results = [77.99, 78.11, 77.25, 77.09, 79.34, 79.03]#, 35.13]

operations = ["concatenate", "add", "wadd", "mul", "subtract", "wsubtract"]#, "circular convolution"]
plt.axis([0, 7, 70,80])

plt.bar(results_no, results, align='center')
plt.xticks(results_no, operations)
# plt.xlabel('dropout')
plt.ylabel('accuracy')
plt.show()



target.close()
