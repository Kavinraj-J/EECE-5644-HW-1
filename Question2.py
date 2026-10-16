import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

#******************* Question 2 Pre-Setup **********************

probL1 = 0.3
probL2 = 0.3
probL3 = 0.4
priors = np.array([probL1, probL2, probL3])

mean1 = np.array([1, 1, 1])
covariance1 = np.array([[1, 0.3, 0],
                        [0.3, 0.8, 0.2],
                        [0, 0.2, 1.2]])
 
mean2 = np.array([1, -1, -1])
covariance2 = np.array([[0.9, -0.2, 0.1],
                        [-0.2, 1.1, 0],
                        [0.1, 0, 0.8]])

mean3a = np.array([-1, 1, -1])
covariance3a = np.array([[1.2, 0, 0.3],
                         [0, 0.9, -0.1],
                         [0.3, -0.1, 1]])
 
mean3b = np.array([-1, -1, 1])
covariance3b = np.array([[0.8, 0.1, 0],
                         [0.1, 1, 0.2],
                         [0, 0.2, 1.1]])
 
sampleSize = 10000

#******************* Question 2 Part A.1 **********************

rng = np.random.default_rng(seed=5644)
sampleData = np.zeros((sampleSize, 3))
 
labels = rng.choice([1, 2, 3], size=sampleSize, p=[probL1, probL2, probL3])
 
for i in range(sampleSize):
    if labels[i] == 1:
        sampleData[i] = rng.multivariate_normal(mean1, covariance1)
    elif labels[i] == 2:
        sampleData[i] = rng.multivariate_normal(mean2, covariance2)
    elif labels[i] == 3:
        component = rng.choice([0, 1], p=[0.5, 0.5])
        if component == 0:
            sampleData[i] = rng.multivariate_normal(mean3a, covariance3a)
        else:
            sampleData[i] = rng.multivariate_normal(mean3b, covariance3b)
 
numL = np.array([np.sum(labels == 1), np.sum(labels == 2), np.sum(labels == 3)])
print("Samples per class (L = 1, 2, 3):", numL)

output = np.column_stack([sampleData, labels])
np.savetxt('EECE-5644-Question2-SampleData.csv', output, delimiter=',', header='x1,x2,x3,L', comments='', fmt=['%.18e'] * 3 + ['%d'])

#******************* Question 2 Part A.2 **********************
likelihood1 = multivariate_normal.pdf(sampleData, mean=mean1, cov=covariance1)
likelihood2 = multivariate_normal.pdf(sampleData, mean=mean2, cov=covariance2)
likelihood3 = (0.5 * multivariate_normal.pdf(sampleData, mean=mean3a, cov=covariance3a)
               + 0.5 * multivariate_normal.pdf(sampleData, mean=mean3b, cov=covariance3b))
likelihoods = np.column_stack([likelihood1, likelihood2, likelihood3])

decisionsMAP = np.argmax(likelihoods * priors, axis=1) + 1

def confusionMatrix(decisions):
    counts = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            counts[i, j] = np.sum((decisions == i + 1) & (labels == j + 1))

    confusion = counts / numL

    return counts, confusion
 
 
countsMAP, confusionMAP = confusionMatrix(decisionsMAP)
print("\nMAP counts #(D=i, L=j):\n", countsMAP)
print("MAP confusion matrix P(D=i | L=j):\n", np.round(confusionMAP, 4))

#******************* Question 2 Part A.3 **********************

def scatterPlot(decisions, title):
    markers = {1: "o", 2: "^", 3: "s"}
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for j in [1, 2, 3]:
        correct = (labels == j) & (decisions == labels)
        incorrect = (labels == j) & (decisions != labels)
        ax.scatter(sampleData[correct, 0], sampleData[correct, 1], sampleData[correct, 2],
                   marker=markers[j], color="green", s=5, label=f"L={j} correct")
        ax.scatter(sampleData[incorrect, 0], sampleData[incorrect, 1], sampleData[incorrect, 2],
                   marker=markers[j], color="red", s=5, label=f"L={j} incorrect")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title(title)
    ax.legend()
    plt.show()
 
 
scatterPlot(decisionsMAP, "MAP classification (0-1 loss)")

#******************* Question 2 Part B **********************

lossMatrix10 = np.array([[0, 1, 10],
                         [1, 0, 10],
                         [1, 1, 0]])
lossMatrix100 = np.array([[0, 1, 100],
                          [1, 0, 100],
                          [1, 1, 0]])
 
for lossMatrix, name in [(lossMatrix10, "Lambda10"), (lossMatrix100, "Lambda100")]:
    risks = (likelihoods * priors) @ lossMatrix.T
    decisionsERM = np.argmin(risks, axis=1) + 1
 
    countsERM, confusionERM = confusionMatrix(decisionsERM)
 
    expectedRisk = np.mean(lossMatrix[decisionsERM - 1, labels - 1])
 
    print(f"\n{name} counts #(D=i, L=j):\n", countsERM)
    print(f"{name} confusion matrix P(D=i | L=j):\n", np.round(confusionERM, 4))
    print(f"{name} minimum expected risk estimate:", expectedRisk)
 
    scatterPlot(decisionsERM, f"ERM classification with {name}")