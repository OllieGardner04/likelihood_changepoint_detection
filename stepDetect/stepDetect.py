import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution

#Class for step detection
class stepDetect:
    """
    This class allows for the detection of changes in the response y, indexed by a covariate x.
    
    """
    
    def __init__(self, x, y, functions, nParams, paramBounds = None):
        #x and y are data
        #functions is a list for the underlying model
        #nParams is the number of parameters within the function
        #paramBounds are the optimisation bounds for the parameter values - NOT theta values
        
        self.x = x
        self.y = y

        self.lambdas = [func for func in functions]

        self.nParams = nParams
        self.nThetas = len(self.lambdas) - 1

        if paramBounds is None: paramBounds = [(-10, 10)]

        if len(paramBounds) == 1: self.paramBounds = paramBounds*self.nParams
        else:                     self.paramBounds = paramBounds

        self.nVars = self.nParams + self.nThetas + 1

    #Function to get mean of response. I.e. g(x)
    def getMu(self, x, params, thetas):
        thetas = np.sort(thetas)
        
        mu = np.zeros_like(x, dtype = np.float64)

        mask0 = x <= thetas[0]

        mu[mask0] = self.lambdas[0](x[mask0], params, thetas)

        for i in range(1, len(thetas)):
            mask = np.logical_and(thetas[i-1] < x, x <= thetas[i])
            
            mu[mask] += self.lambdas[i](x[mask], params, thetas)

        maskFinal = x > thetas[-1]

        mu[maskFinal] += self.lambdas[-1](x[maskFinal], params, thetas)

        return mu

    #Log likelihood of variables given y
    def logLikelihood(self, variables, x, y):
        sigma  = variables[0]
        params = variables[1:self.nParams+1]
        thetas = variables[self.nParams+1:]

        n = len(x)

        mu = self.getMu(x, params, thetas)

        logLik = -n*np.log(sigma) - 1/(2*sigma**2)*np.sum((y - mu)**2)

        return -logLik

    #Calculate MLEs
    def fit(self):
        info = np.iinfo(np.int64)

        self.result = differential_evolution(lambda variables: self.logLikelihood(variables, self.x, self.y),
                                             bounds=[(1e-6,10)] + self.paramBounds + [(0,max(self.x))]*self.nThetas)

        self.resultVars = self.result.x

        self.sigmaHat  = self.resultVars[0]
        self.paramsHat = self.resultVars[1:self.nParams+1]
        self.thetasHat = self.resultVars[self.nParams+1:]

        self.thetasHat = np.sort(self.thetasHat)

        return self.result

    #Plot data and estimated values
    def plot(self):
        plt.scatter(self.x, self.y, color = "black", alpha = 0.2, label = "Data")

        muHat = self.getMu(self.x, self.paramsHat, self.thetasHat)

        plt.plot(self.x, muHat, color = "red", label = "Estimated underlying process")

        plt.plot(self.x, muHat - 1.96*self.sigmaHat, color = "orange", label = "95% C.I.")
        plt.plot(self.x, muHat + 1.96*self.sigmaHat, color = "orange")
        plt.fill_between(self.x, muHat - 1.96*self.sigmaHat, muHat + 1.96*self.sigmaHat, color = "orange", alpha = 0.25)

        plt.axvline(self.thetasHat[0], color = "blue", linestyle = "--", label = "Estimated changepoints")

        for i in self.thetasHat[1:]:
            plt.axvline(i, color = "blue", linestyle = "--")

        plt.legend()

        plt.show()

    #Compare estimated parameters with (observed/true/etc.) parameters
    def compare(self, sigma = None, params = None, thetas = None):

        names  = []
        estims = []
        comps  = []
        
        if sigma is not None:
            names.append("sigma")
            estims.append(self.sigmaHat)
            comps.append(sigma)

        if params is not None:
            for i in range(self.nParams):
                names.append(f"param[{i}]")
                estims.append(self.paramsHat[i])
                comps.append(params[i])

        if thetas is not None:
            for i in range(self.nThetas):
                names.append(f"theta[{i}]")
                estims.append(self.thetasHat[i])
                comps.append(thetas[i])

        compDf = pd.DataFrame({"Name":      names,
                               "Estimated": estims,
                               "Compared":  comps})

        compDf["Bias"] = compDf["Estimated"] - compDf["Compared"]

        return compDf
