import numpy as np
import scipy as spi
import matplotlib.pyplot as plt

class Metropolis: 
    def __init__(self, logTarget, initialState, stepSize=1.0):
        self.logTarget = logTarget
        self.state = initialState
        self.samples = []
        self.stepSize = stepSize
    
    def _accept(self, proposal):
        acceptance_prob = min(0, (self.logTarget(proposal) - self.logTarget(self.state)))
        if np.log(np.random.uniform()) < acceptance_prob:
            self.state = proposal
            return True
        return False
    
    def adapt(self, blockLengths):
        for k in blockLengths:
            accepted = 0
            proposals = 0
            for n in range(k):
                proposal = np.random.normal(self.state, self.stepSize)
                if self._accept(proposal):
                    accepted += 1
                proposals += 1
            acceptance_prob = accepted / proposals
            if acceptance_prob > 0.5:
                self.stepSize *= 1.1
            else:
                self.stepSize *= 0.9
        return self

    def sample(self, nSamples):
        for n in range(nSamples):
            proposal = np.random.normal(loc=self.state, scale=self.stepSize)
            self._accept(proposal)
            self.samples.append(self.state)
        return self

    def summary(self):
        n = len(self.samples)
        mean = np.mean(self.samples)
        std = np.std(self.samples, ddof = 1)
        c025 = np.percentile(self.samples, 2.5)
        c975 = np.percentile(self.samples, 97.5)
        return {'mean': mean, 'std': std, 'c025': c025, 'c975': c975}
    
class SignalDetection:
    def __init__(self, hits, misses, false_alarms, correct_rejections):
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections
    
    def hit_rate(self):
        return (self.hits / (self.hits + self.misses))

    def false_alarm_rate(self):
        return (self.false_alarms / (self.false_alarms + self.correct_rejections))

    def d_prime(self):
        return (spi.stats.norm.ppf(self.hit_rate()) - spi.stats.norm.ppf(self.false_alarm_rate()))

    def criterion(self):
        return -0.5 * (spi.stats.norm.ppf(self.hit_rate()) + spi.stats.norm.ppf(self.false_alarm_rate()))
    
    def __add__(self, other):
        return SignalDetection(self.hits + other.hits, self.misses + other.misses, self.false_alarms + other.false_alarms, self.correct_rejections + other.correct_rejections)
    
    def __mul__(self, scalar):
        return SignalDetection(self.hits * scalar, self.misses * scalar, self.false_alarms * scalar, self.correct_rejections * scalar)
    
    @staticmethod 
    def simulate(dprime, criteriaList, signalCount, noiseCount):
        sdtList = []
        for i in range(len(criteriaList)):
            criterion = criteriaList[i]
            k = criterion + (dprime/2)
            hit_rate = 1 - spi.stats.norm.cdf(k - dprime)
            false_alarm_rate = 1 - spi.stats.norm.cdf(k)
            hits = np.random.binomial(signalCount, hit_rate)
            misses = signalCount - hits
            false_alarms = np.random.binomial(noiseCount, false_alarm_rate)
            correct_rejections = noiseCount - false_alarms
            sdtList.append(SignalDetection(hits, misses, false_alarms, correct_rejections))
        return sdtList
    
    @staticmethod
    def plot_roc(sdtList):
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("Receiver Operating Characteristic Curve")
        if isinstance(sdtList, list):
            for i in range(len(sdtList)):
                sdt = sdtList[i]
                plt.plot(sdt.false_alarm_rate(), sdt.hit_rate(), 'o', color = 'black')
        x, y = np.linspace(0,1,100), np.linspace(0,1,100)
        plt.plot(x,y, '--', color = 'black')
        plt.grid()

    def plot_sdt(self, d_prime):
        x = np.linspace(-4, 4, 1000)
        y_N = spi.stats.norm.pdf(x, loc = 0, scale = 1) 
        y_S = spi.stats.norm.pdf(x, loc = d_prime, scale = 1) 
        c = d_prime/2 #optimal threshold
        Ntop_y = np.max(y_N)
        Nstop_x = x[np.argmax(y_N)]
        Stop_y = np.max(y_S)
        Stop_x = x[np.argmax(y_S)]

    def nLogLikelihood(self, hit_rate, false_alarm_rate):
        return -((self.hits * np.log(hit_rate)) + (self.misses * np.log(1-hit_rate)) + (self.false_alarms * np.log(false_alarm_rate)) + (self.correct_rejections * np.log(1-false_alarm_rate)))

    @staticmethod
    def rocCurve(falseAlarmRate, a):
        return spi.stats.norm.cdf(a + spi.stats.norm.ppf((falseAlarmRate)))
    
    @staticmethod
    def fit_roc(sdtList):
        SignalDetection.plot_roc(sdtList)
        a = 0
        lossfun = spi.optimize.minimize(fun = SignalDetection.rocLoss, x0 = a, method = 'BFGS', args = (sdtList,))
        loss = []
        for i in range(0,100,1):
            loss.append((SignalDetection.rocCurve(i/100, float(lossfun.x))))
        plt.plot(np.linspace(0,1,100), loss, '-', color = 'r')
        aHat = lossfun.x
        return float(aHat)
    
    @staticmethod
    def rocLoss(a, sdtList):
        total_loss = []
        for i in range(len(sdtList)):
            sdt = sdtList[i]
            predicted_hit_rate = sdt.rocCurve(sdt.false_alarm_rate(), a)
            total_loss.append(sdt.nLogLikelihood(predicted_hit_rate, sdt.false_alarm_rate()))
        return sum(total_loss)
    
import unittest

class TestSignalDetection(unittest.TestCase):
    def test_simulate(self):
        # Test with a single criterion value
        dPrime       = 1.5
        criteriaList = [0]
        signalCount  = 1000
        noiseCount   = 1000

        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 1)
        sdt = sdtList[0]

        self.assertEqual(sdt.hits             , sdtList[0].hits)
        self.assertEqual(sdt.misses           , sdtList[0].misses)
        self.assertEqual(sdt.false_alarms      , sdtList[0].false_alarms)
        self.assertEqual(sdt.correct_rejections, sdtList[0].correct_rejections)

        # Test with multiple criterion values
        dPrime       = 1.5
        criteriaList = [-0.5, 0, 0.5]
        signalCount  = 1000
        noiseCount   = 1000
        sdtList      = SignalDetection.simulate(dPrime, criteriaList, signalCount, noiseCount)
        self.assertEqual(len(sdtList), 3)
        for sdt in sdtList:
            self.assertLessEqual    (sdt.hits              ,  signalCount)
            self.assertLessEqual    (sdt.misses            ,  signalCount)
            self.assertLessEqual    (sdt.false_alarms       ,  noiseCount)
            self.assertLessEqual    (sdt.correct_rejections ,  noiseCount)

    def test_nLogLikelihood(self):
        sdt = SignalDetection(10, 5, 3, 12)
        hit_rate = 0.5
        false_alarm_rate = 0.2
        expected_nll = - (10 * np.log(hit_rate) +
                           5 * np.log(1-hit_rate) +
                           3 * np.log(false_alarm_rate) +
                          12 * np.log(1-false_alarm_rate))
        self.assertAlmostEqual(sdt.nLogLikelihood(hit_rate, false_alarm_rate),
                               expected_nll, places=6)

    def test_rocLoss(self):
        sdtList = [
            SignalDetection( 8, 2, 1, 9),
            SignalDetection(14, 1, 2, 8),
            SignalDetection(10, 3, 1, 9),
            SignalDetection(11, 2, 2, 8),
        ]
        a = 0
        expected = 99.3884206555698
        self.assertAlmostEqual(SignalDetection.rocLoss(a, sdtList), expected, places=4)

    def test_integration(self):
        dPrime = 1
        sdtList = SignalDetection.simulate(dPrime, [-1, 0, 1], 1e7, 1e7)
        aHat = SignalDetection.fit_roc(sdtList)
        self.assertAlmostEqual(aHat, dPrime, places=2)
        plt.close()

if __name__ == '__main__':
    unittest.main() 

import scipy.stats

def fit_roc_bayesian(sdtList):

    # Define the log-likelihood function to optimize
    def loglik(a):
        return -SignalDetection.rocLoss(a, sdtList) + scipy.stats.norm.logpdf(a, loc = 0, scale = 10)

    # Create a Metropolis sampler object and adapt it to the target distribution
    sampler = Metropolis(logTarget = loglik, initialState = 0)
    sampler = sampler.adapt(blockLengths = [2000]*3)

    # Sample from the target distribution
    sampler = sampler.sample(nSamples = 4000)

    # Compute the summary statistics of the samples
    result  = sampler.summary()

    # Print the estimated value of the parameter a and its credible interval
    print(f"Estimated a: {result['mean']} ({result['c025']}, {result['c975']})")

    # Create a mosaic plot with four subplots
    fig, axes = plt.subplot_mosaic(
        [["ROC curve", "ROC curve", "traceplot"],
         ["ROC curve", "ROC curve", "histogram"]],
        constrained_layout = True
    )

    # Plot the ROC curve of the SDT data
    plt.sca(axes["ROC curve"])
    SignalDetection.plot_roc(sdtList = sdtList)

    # Compute the ROC curve for the estimated value of a and plot it
    xaxis = np.arange(start = 0.00,
                      stop  = 1.00,
                      step  = 0.01)

    plt.plot(xaxis, SignalDetection.rocCurve(xaxis, result['mean']), 'r-')

    # Shade the area between the lower and upper bounds of the credible interval
    plt.fill_between(x  = xaxis,
                     y1 = SignalDetection.rocCurve(xaxis, result['c025']),
                     y2 = SignalDetection.rocCurve(xaxis, result['c975']),
                     facecolor = 'r',
                     alpha     = 0.1)

    # Plot the trace of the sampler
    plt.sca(axes["traceplot"])
    plt.plot(sampler.samples)
    plt.xlabel('iteration')
    plt.ylabel('a')
    plt.title('Trace plot')

    # Plot the histogram of the samples
    plt.sca(axes["histogram"])
    plt.hist(sampler.samples,
             bins    = 51,
             density = True)
    plt.xlabel('a')
    plt.ylabel('density')
    plt.title('Histogram')

    # Show the plot
    plt.show()

# Define the number of SDT trials and generate a simulated dataset
sdtList = SignalDetection.simulate(dprime       = 1,
                                   criteriaList = [-1, 0, 1],
                                   signalCount  = 40,
                                   noiseCount   = 40)

# Fit the ROC curve to the simulated dataset
fit_roc_bayesian(sdtList)