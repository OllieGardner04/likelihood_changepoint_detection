import numpy  as np
import pandas as pd

from stepDetect import stepDetect

def constantsExample():
    x = np.arange(0, 1000)

    y = np.zeros_like(x, dtype = np.float64)

    sigma  = 0.5

    g0 = lambda x, params, thetas: params[0]
    g1 = lambda x, params, thetas: params[1]
    g2 = lambda x, params, thetas: params[2]
    g3 = lambda x, params, thetas: params[3]
    g4 = lambda x, params, thetas: params[4]

    paramVals = [10,  2,   8,   1,   9]
    thetaVals = [350, 400, 700, 800, 1000]

    y[x <= thetaVals[-1]] = paramVals[-1]
    y[x <= thetaVals[-2]] = paramVals[-2]
    y[x <= thetaVals[-3]] = paramVals[-3]
    y[x <= thetaVals[-4]] = paramVals[-4]
    y[x <= thetaVals[-5]] = paramVals[-5]

    y += np.random.normal(0, sigma, size = len(x))

    s = stepDetect(x, y, [g0, g1, g2, g3, g4], 5)

    res = s.fit()

    print(s.compare(sigma, paramVals, thetaVals))

    s.plot()

constantsExample()
