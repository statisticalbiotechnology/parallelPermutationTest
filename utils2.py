import numpy as np
import concurrent.futures as cf
import matplotlib.pyplot as plt
import seaborn as sns

def score_distribution_numpy_full(digitized,K,S,L, data_type=np.float64):
    # N(s,l) number of ways to reach a sum of s using k of the l first readouts
    # Calculated by iterating over the
    N = np.zeros((S,L,K), dtype=data_type)
    # Initiating (i.e. k=0 case)
    for l in range(L):
        d=digitized[l]
        N[d,l,0] = data_type(1)
    # Do each of the other picks
    for k in range(1,K):
        C = np.zeros((S,1), dtype=data_type)
        for l in range(1,L): #hoehle: why 1:L and not 0:L?
            d = digitized[l]
            C +=  N[:,l-1:l,k-1]
            N[d:S,l,k] = C[0:S-d,0]
    NN = np.sum(N[:,:,-1],axis=1)
    return NN

def significance_of_mean(a,b,num_bin = None, data_type=np.float64): #

    # discretize
    ab = np.sort(np.concatenate((a, b), axis=None))[::-1] #sort descending (biggest first)
    
    if not num_bin:
        num_bin = np.ceil(max(ab)) - np.floor(min(ab)) + 1
    
    

    bins = np.linspace(min(ab), max(ab), num_bin)    
    #bins = np.linspace(np.floor(min(ab)), np.ceil(max(ab)), num_bin)
    digitized = np.digitize(ab, bins)
    
    if len(a)>len(b):
        score = sum(np.digitize(b,bins))
    else:
        score = sum(np.digitize(a,bins))
    K = min(len(a),len(b))
    S = np.sum(digitized[0:K])+1 #hoehle: This is the maximum sum we can get! why +1? A: S is the number of bins, lowest score=0, highest score=sum(digitiyzed) make a total of sum(digitiyzed)+1 bins
    L = len(ab) #length of m+n
    NN = score_distribution_numpy_full(digitized, K, S, L, data_type)
    # Total number of configurations
    total = np.sum(NN)
    # Just the one sided version of the test, i.e. alternative="greater"
    if len(a)>len(b):
        more_extreme = np.sum(NN[:score])
    else:
        more_extreme = np.sum(NN[score:])
    p = more_extreme/total
    return p, total, more_extreme

def p_value_calc(args):
    """Get p-values single-thread"""
    a,b, bins = args
    p=significance_of_mean(a,b, bins)[0]
    return p

def calibration_series_generator(A,B, S):
    """Yield samples"""
    num_tests = A.shape[0]
    for i in range(num_tests):
        a_sample = A[i].tolist()
        b_sample = B[i].tolist()
        yield ([a_sample,b_sample, S])

def calibration_test(A,B,bins):
    """Run single-thread Green"""
    with cf.ProcessPoolExecutor(max_workers=1) as pool:
        p_list = list(pool.map(p_value_calc, calibration_series_generator(A,B, bins)))
    return p_list
    
def preparePandas(timeData, sampleSizes, test="Parallel"):
    """Create DF for sns-plots"""
    preparePd = list()
    for time, sample in zip(timeData, sampleSizes):
        preparePd.append([str(test),time, sample])
        
            
    return preparePd

def timePlotSNS(TIMEParallel, TIMEsingleThred, sampleShape,binVar=False, log=False, path=None):
    """Create SNS timeseries-plot"""
    fig, ax = plt.subplots()
    
    sns.set(style="white")
    sns.set_context("talk")
    
    preparePdParallel = preparePandas(TIMEParallel, sampleShape)
    preparePdSingle = preparePandas(TIMEsingleThred, sampleShape, 'Single thread')
    
    data = preparePdParallel + preparePdSingle
    
    pdData = pd.DataFrame(data, columns=['Method', 'time(s)','bins'])
    
    if log:        
        MAX = max(max(TIMEParallel), max(TIMEsingleThred))
        MIN = min(min(TIMEParallel), min(TIMEsingleThred))

        RANGE = np.arange(np.floor(MIN), np.ceil(MAX))
        snsPlot = sns.lineplot(x="bins", y="time(s)",
             hue="Method",
             data=pdData)#.set(yticks = RANGE, yticklabels=10**RANGE)
        plt.yticks(RANGE, 10.0**RANGE)
        
        
    else:
        snsPlot = sns.lineplot(x="bins", y="time(s)",
             hue="Method",
             data=pdData,)
    
    if binVar:
        plt.xlabel(r"$n$")
        
    else:
        plt.xlabel(r"$n_{w}$")
    
    
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    
    plt.setp(snsPlot.get_legend().get_texts(), fontsize='12')
    plt.tight_layout()
    
    if path:   
        fig = snsPlot.get_figure()
        fig.savefig(path)

def getPATH(path, name):
    """Get path for figures"""
    return path + '/'+ name 