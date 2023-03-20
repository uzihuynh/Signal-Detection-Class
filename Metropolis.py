import numpy as np
import scipy.stats

class Metropolis:
    def __init__(self, logTarget, initialState):
        self.logTarget = logTarget
        self.currentState = initialState
        self.sigma = 1
        self.samples = []

    def accept(self, proposal):
        acceptance_prob = min(1, np.exp(self.logTarget(proposal) - self.logTarget(self.currentState)))
        return np.random.uniform() < acceptance_prob

    def adapt(self, blockLengths):
        target_acceptance_rate = 0.4

        for block_length in blockLengths:
            accepted_count = 0

            for _ in range(block_length):
                proposal = np.random.normal(loc=self.currentState, scale=self.sigma)
                if self.accept(proposal):
                    self.currentState = proposal
                    accepted_count += 1

            acceptance_rate = accepted_count / block_length
            self.sigma *= target_acceptance_rate / acceptance_rate

        return self

    def sample(self, n):
        for _ in range(n):
            proposal = np.random.normal(loc=self.currentState, scale=self.sigma)
            if self.accept(proposal):
                self.currentState = proposal
            self.samples.append(self.currentState)

        return self

    def summary(self):
        samples_np = np.array(self.samples)
        mean = np.mean(samples_np)
        c025, c975 = np.percentile(samples_np, [2.5, 97.5])

        return {'mean': mean, 'c025': c025, 'c975': c975}

import scipy as spi
import matplotlib.pyplot as plt

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
        plt.figure()
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

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

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