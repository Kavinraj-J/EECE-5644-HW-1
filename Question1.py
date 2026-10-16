import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.linalg import eigh

#************** Question 1 Pre Setup ********************

fileName = 'EECE-5644-Question1-SampleData.csv';

probL0 = 0.35
probL1 = 0.65

mean0 = np.array([-0.5, -0.5, -0.5, -0.5])
covariance0 = 0.25 * np.array([[2, -0.5, 0.3, 0],
               [-0.5, 1, -0.5, 0],
               [0.3, -0.5, 1, 0],
               [0, 0, 0, 2]])

mean1 = np.array([1, 1, 1, 1])
covariance1 = np.array([[1, 0.3, -0.2, 0],
               [0.3, 2, 0.3, 0],
               [-0.2, 0.3, 1, 0],
               [0, 0, 0, 3]])

sampleSize = 10000


rng = np.random.default_rng(seed=5644)
sampleData = np.zeros((sampleSize, 4))

choices = rng.choice([0, 1], size=sampleSize, p=[probL0, probL1])

print("L = 0: ", np.sum(choices == 0))
print("L = 1: ", np.sum(choices == 1))

for i in range(sampleSize):
    if(choices[i] == 0):
        sampleData[i] = rng.multivariate_normal(mean0, covariance0)
    elif(choices[i] == 1):
        sampleData[i] = rng.multivariate_normal(mean1, covariance1)

output = np.column_stack([sampleData, choices])
np.savetxt(fileName, output, delimiter=',', header='x1,x2,x3,x4,L', comments='', fmt=['%.18e'] * 4 + ['%d'])

#************** Question 1 Part A.2 ********************

numL0 = np.sum(choices == 0)
numL1 = np.sum(choices == 1)

likelihood1 = multivariate_normal.pdf(sampleData, mean=mean1, cov=covariance1)
likelihood0 = multivariate_normal.pdf(sampleData, mean=mean0, cov=covariance0)
likelihoodRatio = likelihood1 / likelihood0

gammas = np.concatenate(([0], np.sort(likelihoodRatio), [np.inf]))

probTruePositive = np.zeros(len(gammas))    # P(D=1 | L=1; gamma)
probFalsePositive = np.zeros(len(gammas))   # P(D=1 | L=0; gamma)
probFalseNegative = np.zeros(len(gammas))   # P(D=0 | L=1; gamma)

for k in range(len(gammas)):
    decisions = (likelihoodRatio > gammas[k]).astype(int)
    probTruePositive[k] = np.sum((decisions == 1) & (choices == 1)) / numL1
    probFalsePositive[k] = np.sum((decisions == 1) & (choices == 0)) / numL0
    probFalseNegative[k] = np.sum((decisions == 0) & (choices == 1)) / numL1

plt.plot(probFalsePositive, probTruePositive)
plt.xlabel("False positive probability  P(D=1 | L=0)")
plt.ylabel("True positive probability  P(D=1 | L=1)")
plt.title("ROC curve of the minimum expected risk classifier")
# plt.show()

#************** Question 1 Part A.3 ********************
probError = probFalsePositive * probL0 + probFalseNegative * probL1

minIndex = np.argmin(probError)
gammaEmpirical = gammas[minIndex]
minProbError = probError[minIndex]

plt.plot(probFalsePositive[minIndex], probTruePositive[minIndex], "ro",
         label="Minimum P(error)")
plt.legend()
plt.show()

print("Empirical gamma (min P(error)):", gammaEmpirical)
print("P(D=1 | L=1) at this gamma:", probTruePositive[minIndex])
print("P(D=1 | L=0) at this gamma:", probFalsePositive[minIndex])
print("Minimum P(error) estimate:", minProbError)


#************** Question 1 Part B ********************

covarianceNB0 = np.diag(np.diag(covariance0))
covarianceNB1 = np.diag(np.diag(covariance1))

likelihoodNB1 = multivariate_normal.pdf(sampleData, mean=mean1, cov=covarianceNB1)
likelihoodNB0 = multivariate_normal.pdf(sampleData, mean=mean0, cov=covarianceNB0)
likelihoodRatioNB = likelihoodNB1 / likelihoodNB0

gammasNB = np.concatenate(([0], np.sort(likelihoodRatioNB), [np.inf]))

probTruePositiveNB = np.zeros(len(gammasNB))    # P(D=1 | L=1; gamma)
probFalsePositiveNB = np.zeros(len(gammasNB))   # P(D=1 | L=0; gamma)
probFalseNegativeNB = np.zeros(len(gammasNB))   # P(D=0 | L=1; gamma)

for k in range(len(gammasNB)):
    decisions = (likelihoodRatioNB > gammasNB[k]).astype(int)
    probTruePositiveNB[k] = np.sum((decisions == 1) & (choices == 1)) / numL1
    probFalsePositiveNB[k] = np.sum((decisions == 1) & (choices == 0)) / numL0
    probFalseNegativeNB[k] = np.sum((decisions == 0) & (choices == 1)) / numL1

probErrorNB = probFalsePositiveNB * probL0 + probFalseNegativeNB * probL1

minIndexNB = np.argmin(probErrorNB)
gammaEmpiricalNB = gammasNB[minIndexNB]
minProbErrorNB = probErrorNB[minIndexNB]

plt.figure()
plt.plot(probFalsePositiveNB, probTruePositiveNB, label="ROC curve")
plt.plot(probFalsePositiveNB[minIndexNB], probTruePositiveNB[minIndexNB], "ro",
         label="Minimum P(error)")
plt.xlabel("False positive probability  P(D=1 | L=0)")
plt.ylabel("True positive probability  P(D=1 | L=1)")
plt.title("ROC curve of the Naive Bayesian classifier")
plt.legend()
plt.show()

print("***************")
print("Naive Bayes empirical gamma (min P(error)):", gammaEmpiricalNB)
print("Naive Bayes P(D=1 | L=1) at this gamma:", probTruePositiveNB[minIndexNB])
print("Naive Bayes P(D=1 | L=0) at this gamma:", probFalsePositiveNB[minIndexNB])
print("Naive Bayes minimum P(error) estimate:", minProbErrorNB)

#************** Question 1 Part C ********************
sampleMean0 = np.mean(sampleData[choices == 0], axis=0)
sampleMean1 = np.mean(sampleData[choices == 1], axis=0)
sampleCovariance0 = np.cov(sampleData[choices == 0], rowvar=False, bias=True)
sampleCovariance1 = np.cov(sampleData[choices == 1], rowvar=False, bias=True)

scatterBetween = np.outer(sampleMean1 - sampleMean0, sampleMean1 - sampleMean0)
scatterWithin = sampleCovariance0 + sampleCovariance1

eigenvalues, eigenvectors = eigh(scatterBetween, scatterWithin)
wLDA = eigenvectors[:, np.argmax(eigenvalues)]
if wLDA @ (sampleMean1 - sampleMean0) < 0:
    wLDA = -wLDA

projection = sampleData @ wLDA
taus = np.concatenate(([-np.inf], np.sort(projection), [np.inf]))

probTruePositiveLDA = np.zeros(len(taus))    # P(D=1 | L=1; tau)
probFalsePositiveLDA = np.zeros(len(taus))   # P(D=1 | L=0; tau)
probErrorLDA = np.zeros(len(taus))           # P(error; tau) from sample counts

for k in range(len(taus)):
    decisions = (projection > taus[k]).astype(int)
    probTruePositiveLDA[k] = np.sum((decisions == 1) & (choices == 1)) / numL1
    probFalsePositiveLDA[k] = np.sum((decisions == 1) & (choices == 0)) / numL0
    probErrorLDA[k] = np.sum(decisions != choices) / sampleSize

minIndexLDA = np.argmin(probErrorLDA)
tauBest = taus[minIndexLDA]
minProbErrorLDA = probErrorLDA[minIndexLDA]

plt.figure()
plt.plot(probFalsePositiveLDA, probTruePositiveLDA, label="ROC curve")
plt.plot(probFalsePositiveLDA[minIndexLDA], probTruePositiveLDA[minIndexLDA], "ro",
         label="Minimum P(error)")
plt.xlabel("False positive probability  P(D=1 | L=0)")
plt.ylabel("True positive probability  P(D=1 | L=1)")
plt.title("ROC curve of the Fisher LDA classifier")
plt.legend()
plt.show()

print("***************")
print("Estimated mean, class 0:", sampleMean0)
print("Estimated mean, class 1:", sampleMean1)
print("Estimated covariance, class 0:\n", sampleCovariance0)
print("Estimated covariance, class 1:\n", sampleCovariance1)
print("w_LDA:", wLDA)
print("LDA best tau (min P(error)):", tauBest)
print("LDA P(D=1 | L=1) at this tau:", probTruePositiveLDA[minIndexLDA])
print("LDA P(D=1 | L=0) at this tau:", probFalsePositiveLDA[minIndexLDA])
print("LDA minimum P(error) estimate:", minProbErrorLDA)