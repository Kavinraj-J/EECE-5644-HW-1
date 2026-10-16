import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def estimateParameters(X, L):
    classes = np.unique(L)
    priors, means, covariances = {}, {}, {}
    print("\nClass | N_j | prior | rank(C) | condition number")
    for c in classes:
        Xc = X[L == c]
        priors[c] = len(Xc) / len(X)
        means[c] = np.mean(Xc, axis=0)
        covariances[c] = np.cov(Xc, rowvar=False, bias=True)
        eigenvalues = np.linalg.eigvalsh(covariances[c])
        rank = np.linalg.matrix_rank(covariances[c])
        conditionNumber = eigenvalues.max() / eigenvalues.min() if rank == X.shape[1] else np.inf
        print(c, len(Xc), round(priors[c], 4), rank, "%.2e" % conditionNumber)
    return classes, priors, means, covariances

def regularizationLambda(C, alpha, mode):
    rank = np.linalg.matrix_rank(C)
    nonzeroEigenvalues = np.sort(np.linalg.eigvalsh(C))[::-1][:rank]
    if mode == "arithmetic":
        return alpha * np.trace(C) / rank
    else:
        return alpha * np.exp(np.mean(np.log(nonzeroEigenvalues)))

def mapClassify(X, classes, priors, means, covariances, alpha, mode):
    scores = np.zeros((len(X), len(classes)))
    lambdas = {}
    for k, c in enumerate(classes):
        lambdas[int(c)] = regularizationLambda(covariances[c], alpha, mode)
        covarianceReg = covariances[c] + lambdas[int(c)] * np.eye(X.shape[1])
        scores[:, k] = multivariate_normal.logpdf(X, mean=means[c], cov=covarianceReg) + np.log(priors[c])
    return classes[np.argmax(scores, axis=1)], lambdas

def compareRegularization(X, L, classes, priors, means, covariances):
    print("\nalpha | P(error) arithmetic-mean lambda | P(error) geometric-mean lambda")
    for alpha in [0.0001, 0.001, 0.01, 0.1]:
        errArith = np.mean(mapClassify(X, classes, priors, means, covariances, alpha, "arithmetic")[0] != L)
        errGeo = np.mean(mapClassify(X, classes, priors, means, covariances, alpha, "geometric")[0] != L)
        print(alpha, round(errArith, 4), round(errGeo, 4))

def confusionMatrix(decisions, L, classes):
    confusion = np.zeros((len(classes), len(classes)))
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            confusion[i, j] = np.sum((decisions == ci) & (L == cj)) / np.sum(L == cj)
    return confusion

def skewness(X):
    return np.mean(((X - X.mean(0)) / X.std(0)) ** 3, axis=0)

def pcaPlots(X, L, classes, classNames, title, colors):
    standardized = (X - X.mean(0)) / X.std(0)
    U, S, Vt = np.linalg.svd(standardized, full_matrices=False)
    pcs = standardized @ Vt[:3].T
    explained = S ** 2 / np.sum(S ** 2)
    print("Variance explained by PC1-PC3:", np.round(explained[:3], 3))
 
    plt.figure()
    for k, c in enumerate(classes):
        plt.scatter(pcs[L == c, 0], pcs[L == c, 1], s=4, color=colors[k], label=classNames[k])
    plt.xlabel(f"PC1 ({explained[0]:.0%} of variance)")
    plt.ylabel(f"PC2 ({explained[1]:.0%} of variance)")
    plt.title(f"{title}: first two principal components")
    plt.legend(markerscale=3)
    plt.show()
 
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for k, c in enumerate(classes):
        ax.scatter(pcs[L == c, 0], pcs[L == c, 1], pcs[L == c, 2], s=3, color=colors[k], label=classNames[k])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"{title}: first three principal components")
    ax.legend(markerscale=3)
    plt.show()


#****************** Wine Quality **********************
wineData = np.loadtxt('winequality-white.csv', delimiter=';', skiprows=1)
wineX = wineData[:, :11]
wineL = wineData[:, 11].astype(int)
featureNames = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
                "pH", "sulphates", "alcohol"]
print("WINE  Samples:", wineX.shape[0], " Features:", wineX.shape[1], " Classes present:", np.unique(wineL))
 
wineClasses, winePriors, wineMeans, wineCovariances = estimateParameters(wineX, wineL)
compareRegularization(wineX, wineL, wineClasses, winePriors, wineMeans, wineCovariances)
 
wineDecisions, wineLambdas = mapClassify(wineX, wineClasses, winePriors, wineMeans, wineCovariances, 0.01, "geometric")
print("\nlambda per class:", {c: float("%.3g" % v) for c, v in wineLambdas.items()})
wineErrors = np.sum(wineDecisions != wineL)
print("Errors:", wineErrors, " P(error) estimate:", round(wineErrors / len(wineL), 4))
print("\nConfusion matrix P(D=i | L=j), rows/columns =", wineClasses)
print(np.round(confusionMatrix(wineDecisions, wineL, wineClasses), 3))
 
print("\nFeature skewness:")
for name, s in zip(featureNames, skewness(wineX)):
    print(f"  {name}: {s:.2f}")
 
pcaPlots(wineX, wineL, wineClasses, [f"quality {c}" for c in wineClasses], "White wine",
         plt.cm.viridis(np.linspace(0, 1, len(wineClasses))))
 
plt.figure()
colors = plt.cm.viridis(np.linspace(0, 1, len(wineClasses)))
for k, c in enumerate(wineClasses):
    plt.scatter(wineX[wineL == c, 10], wineX[wineL == c, 7], s=4, color=colors[k], label=f"quality {c}")
plt.xlabel("alcohol")
plt.ylabel("density")
plt.ylim(0.985, 1.005)
plt.title("White wine: alcohol vs density")
plt.legend(markerscale=3)
plt.show()


#****************** Human Activity Recognition **********************
harX = np.vstack([np.loadtxt('X_train.txt'),
                  np.loadtxt('X_test.txt')])
harL = np.concatenate([np.loadtxt('y_train.txt'),
                       np.loadtxt('y_test.txt')]).astype(int)
activityNames = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
print("\n\nHAR  Samples:", harX.shape[0], " Features:", harX.shape[1], " Classes present:", np.unique(harL))
 
harClasses, harPriors, harMeans, harCovariances = estimateParameters(harX, harL)
compareRegularization(harX, harL, harClasses, harPriors, harMeans, harCovariances)
 
harDecisions, harLambdas = mapClassify(harX, harClasses, harPriors, harMeans, harCovariances, 0.01, "arithmetic")
print("\nlambda per class:", {c: float("%.3g" % v) for c, v in harLambdas.items()})
harErrors = np.sum(harDecisions != harL)
print("Errors:", harErrors, " P(error) estimate:", round(harErrors / len(harL), 4))
print("\nConfusion matrix P(D=i | L=j), rows/columns =", activityNames)
print(np.round(confusionMatrix(harDecisions, harL, harClasses), 3))
 
harSkewness = skewness(harX)
print("\nFraction of features with |skewness| > 1:", round(np.mean(np.abs(harSkewness) > 1), 3))
print("Median |skewness|:", round(np.median(np.abs(harSkewness)), 2), " Max skewness:", round(harSkewness.max(), 2))
print("Fraction of feature values at the -1 or +1 bound:", round(np.mean(np.abs(harX) >= 0.9999), 3))
 
pcaPlots(harX, harL, harClasses, activityNames, "HAR", plt.cm.tab10(np.arange(len(harClasses))))