import random
import numpy as np
import matplotlib.pyplot as plt

w0 = random.uniform(-1 / 4, 1 / 4)
w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)

actualWeightMatrix = [-0.08563611750071531, -0.3198498484123551, 0.5440033003143925]
print('ActualWeightMatrix')
print(actualWeightMatrix)

np.random.seed(1)
trainingSamples = np.zeros(shape=(100, 2))
trainingSamples = np.array(np.random.uniform(-1, 1, (100, 2)))

n, m = trainingSamples.shape
X0 = np.ones((n, 1))
trainingSamplesNew = np.hstack((X0, trainingSamples))
print(trainingSamplesNew)

outputMatrix = trainingSamplesNew.dot(actualWeightMatrix)
# Initial Classification for desired outputs
# S1 Class elements
count = 0
S1Class = []
for item in outputMatrix:
    if item > 0 or item == 0:
        S1Class.append(int(np.where(outputMatrix == item)[0]))
        # print np.where(outputMatrix == item)[0]

print('S1 Class')
print(S1Class)
S = list(range(100))

# S0 Class of training samples
S0 = list(set(S) - set(S1Class))
print('S0 class')
print(S0)

# S1 and S0 lists hold the indices of the training samples

# Method for extending the separator line
def extended(ax1, x, y, **args):
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()

    x_ext = np.linspace(xlim[0], xlim[1], 100)
    p = np.polyfit(x, y, deg=1)
    y_ext = np.poly1d(p)(x_ext)
    ax1.plot(x_ext, y_ext, **args)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    return ax1

# To plot the S1 class elements
ax = plt.subplot(111)
ax.scatter([row[0] for row in trainingSamples[S1Class]], [row[1] for row in trainingSamples[S1Class]], label="S1 Class",
           color="red", marker="*", s=30)

# To plot the S0 class elements
ax.scatter([row[0] for row in trainingSamples[S0]], [row[1] for row in trainingSamples[S0]], label="S0 Class", color=
"blue", marker="o", s=30)

xAxisPoints = [0, -actualWeightMatrix[0] / actualWeightMatrix[1]]
yAxisPoints = [-actualWeightMatrix[0] / actualWeightMatrix[2], 0]

# To plot the separator line
ax = extended(ax, xAxisPoints, yAxisPoints, color="black", lw=1)
ax.plot(xAxisPoints, yAxisPoints, color="black", lw=1, label="Separator")

ax.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
# End of code snippet for 3i

# Variable for Eta which is replaced by 10 and 0.1 for next questions
eta = 1
w0_new = random.uniform(-1 / 4, 1 / 4)
w1_new = random.uniform(-1, 1)
w2_new = random.uniform(-1, 1)

# initialWeightMatrix = [w0_new, w1_new, w2_new]
initialWeightMatrix = [0.07124425732254824, -0.6306907909877648, 0.04286161458401949]
print('Initial Weight Matrix')
print(initialWeightMatrix)

def intersection(lst1, lst2):
    mCount = 0
    lst3 = [value for value in lst1 if value in lst2]
    if len(lst2) == len(lst3):
        mCount = 0
    else:
        mCount = abs(len(lst3) - len(lst1))
    return mCount


epochNumber = 0
epochList = []
misClassificationList = []
misClassifications = 1
flag = 0

# while misClassifications != 0:
for epochNumber in range(1, 10):
    outputMatrixNew = trainingSamplesNew.dot(initialWeightMatrix)
    count = 0
    S1ClassNew = []
    for item in outputMatrixNew:
        if item > 0:
            S1ClassNew.append(int(np.where(outputMatrixNew == item)[0]))

    misClassifications = intersection(S1ClassNew, S1Class)
    epochList.append(epochNumber)
    misClassificationList.append(misClassifications)

    for i in range(0, len(trainingSamples)):
        tempOutputMatrix = trainingSamplesNew[i].dot(initialWeightMatrix)
        if (tempOutputMatrix > 0 or tempOutputMatrix == 0) and i in S1Class:
            continue
        elif (tempOutputMatrix > 0 or tempOutputMatrix == 0) and i in S0:
            initialWeightMatrix = initialWeightMatrix - eta * trainingSamplesNew[i]
        elif tempOutputMatrix < 0 and i in S1Class:
            initialWeightMatrix = initialWeightMatrix + eta * trainingSamplesNew[i]
        elif tempOutputMatrix < 0 and i in S0:
            continue

print('Final Weight Matrix')
print(initialWeightMatrix)

# Plot for epochnumber vs number of misclassifications
ax1 = plt.subplot(111)
ax1.plot(epochList, misClassificationList, color="black", lw=1)
ax1.legend()
plt.xlabel('EpochNumber')
plt.ylabel('Number of misclassifications')
plt.show()

# Plot to check if the obtained weights separate the samples into classes correctly
ax2 = plt.subplot(111)
ax2.scatter([row[0] for row in trainingSamples[S1Class]], [row[1] for row in trainingSamples[S1Class]],
            label="S1 Class New",
            color="red", marker="*", s=30)

ax2.scatter([row[0] for row in trainingSamples[S0]], [row[1] for row in trainingSamples[S0]], label="S0 Class New",
            color=
            "blue", marker="o", s=30)

xAxisPoints = [0, -initialWeightMatrix[0] / initialWeightMatrix[1]]
yAxisPoints = [-initialWeightMatrix[0] / initialWeightMatrix[2], 0]

ax2 = extended(ax2, xAxisPoints, yAxisPoints, color="black", lw=1)
ax2.plot(xAxisPoints, yAxisPoints, color="black", lw=1, label="Separator")

ax2.legend()
plt.show()



