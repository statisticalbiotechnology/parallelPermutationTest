import numpy as np
import concurrent.futures as cf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from significance_of_mean_cuda import significance_of_mean_cuda
from scipy.stats import ttest_ind, mannwhitneyu
import numpy.random as npr

import numpy as np
from multiprocessing.pool import ThreadPool
import itertools
from multiprocessing import sharedctypes
import multiprocessing 

def getNumeratorCPU(m, n, S, z, dtype, cores=-1):
    if cores == -1:
        cores = multiprocessing.cpu_count()
        
    
    N = np.zeros([S + 1, m], dtype)
    N_old = N.copy()
    
    
    N = np.ctypeslib.as_ctypes(np.zeros((S + 1, m)))
    N_shared_array = sharedctypes.RawArray(N._type_, N)
    
    def multiprocessing_func(xLim, yLim, N_shared_array):
        tmp = np.ctypeslib.as_array(N_shared_array)
    
        underLimX, upperLimX = xLim
        underLimY, upperLimY = yLim

    
        for j in range(underLimY, upperLimY):
            for s in range(underLimX, upperLimX):
                if j >= m + 1 or s > S or j < 1:
                    pass
                elif s == 0 and int(j-1)==0:
                    tmp[s,j-1] =  1
                elif i < j:
                    tmp[s, j - 1] = 0
                elif j > 1 and z[i-1] <= s:
                    tmp[s,j-1] = N_old[s - z[i -1], j-2] + N_old[s,j-1]
                elif j > 1 and z[i-1] > s:
                    tmp[s,j-1] = N_old[s,j-1]
    
    
    x_len, y_len = S + 1, m + 1

    batchsize =  int(x_len / cores)
        
    for i in range(1,(m+n)+1): 
        processes = []
        for underLim in range(0, x_len, batchsize):
            if underLim + batchsize > x_len:
                upperLim = x_len
            else:
                upperLim = underLim + batchsize
                
            p = multiprocessing.Process(target=multiprocessing_func, args=((underLim, upperLim),(0,y_len), N_shared_array))
            processes.append(p)
            p.start()
        
        for process in processes:
            process.join()
        
        N_old = np.ctypeslib.as_array(N_shared_array).copy()
        

    return N_old[:,-1]

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

def preparePandasFastperm(timeData, sampleSizes, test="Parallel", replicates=True):
    preparePd = list()
    for time, sample in zip(timeData, sampleSizes):
        if isinstance(time,float):
            preparePd.append([str(test),time, sample])
        else:
            for t in time:
                preparePd.append([str(test),t, sample])
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
        plt.xlabel(r"$n$", fontsize=20)
        
    else:
        plt.xlabel(r"$n_{w}$", fontsize=20)
    
    plt.ylabel(r"$time(s)$", fontsize=20)
    
    
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    
    plt.setp(snsPlot.get_legend().get_texts(), fontsize=20)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.tight_layout()
    
    if path:   
        fig = snsPlot.get_figure()
        fig.savefig(path)

def getPATH(path, name):
    """Get path for figures"""
    return path + '/' + name
    
def getSynteticData(func, setN=20, sampleN=2_000, mean=0, std=1,seed=1):
    """Generate synthetic data"""
    np.random.seed(seed)
    AN, BN = [func(mean,std,setN) for i in range(sampleN)], [func(0,std,setN) for i in range(sampleN)]
    return AN, BN

def getAllSynthticData(sampleRange, mean,sampleN=50):
    """Get all synthetic data"""
    A_data, B_data = list(), list()
    for setS in sampleRange:
        Anorm0, Bnorm0 = getSynteticData(np.random.normal, mean=mean, setN=setS,sampleN=sampleN)
        A_data.append(Anorm0)
        B_data.append(Bnorm0)
    return A_data, B_data

def timePlotSNSFastperm(TIMEParallel, TIME_MC, sampleShape, log=False, TIMEsingleThred=False, path=None):
    
    sns.set(style="white")
    sns.set_context("talk")
    
    preparePdParallel = preparePandasFastperm(TIMEParallel, sampleShape, "Parallel Green")
    preparePdMc = preparePandasFastperm(TIME_MC, sampleShape, 'FastPerm')
    
    if np.any(TIMEsingleThred):
        preparePdSingle = preparePandasFastperm(TIMEsingleThred, sampleShape, 'Single thread Green')
        data = preparePdMc + preparePdParallel + preparePdSingle
    else:
        data = preparePdMc + preparePdParallel
        
    pdData = pd.DataFrame(data, columns=['Method', 'time(s)','n'])

    
    if log:        
        MAX = max(np.max(TIMEParallel), np.max(TIME_MC))
        MIN = min(np.min(TIMEParallel), np.min(TIME_MC))
        if np.any(TIMEsingleThred):
            MAX = max(np.max(TIMEParallel), np.max(TIME_MC), np.max(TIMEsingleThred))
            MIN = min(np.min(TIMEParallel), np.min(TIME_MC), np.min(TIMEsingleThred))
            
        
        RANGE = np.arange(np.floor(MIN), np.ceil(MAX))
        
        snsPlot = sns.lineplot(x="n", y="time(s)",
             hue="Method",
             data=pdData)
        plt.yticks(RANGE, 10.0**RANGE)
        
        
    else:
        snsPlot = sns.lineplot(x="n", y="time(s)",
             hue="Method",
             data=pdData)
        
        
    h,l = snsPlot.get_legend_handles_labels()
    plt.legend(h[1:],l[1:])

    plt.ylabel("time(s)",fontsize=20)
        
        
    plt.xlabel(r"$n$",fontsize=15)

    plt.tight_layout()
    if path:
        
        fig = snsPlot.get_figure()


        fig.savefig(path)

def GPUTimeComparisonPlot(RTX2060, RTX2070, sampleShape, log=False, TITANX=False, path=None):
    
    sns.set(style="white")
    sns.set_context("talk")
    
    preparePdParallel = preparePandasFastperm(RTX2060, sampleShape, "RTX2060")
    preparePdMc = preparePandasFastperm(RTX2070, sampleShape, 'RTX2070')
    
    if np.any(TITANX):
        preparePdSingle = preparePandasFastperm(TITANX, sampleShape, 'TITANX')
        data = preparePdMc + preparePdParallel + preparePdSingle
    else:
        data = preparePdMc + preparePdParallel
        
    pdData = pd.DataFrame(data, columns=['Method', 'time(s)','n'])

    
    if log:        
        MAX = max(np.max(RTX2060), np.max(RTX2070))
        MIN = min(np.min(RTX2060), np.min(RTX2070))
        if np.any(TITANX):
            MAX = max(np.max(RTX2060), np.max(RTX2070), np.max(TITANX))
            MIN = min(np.min(RTX2060), np.min(RTX2070), np.min(TITANX))
            
        
        RANGE = np.arange(np.floor(MIN), np.ceil(MAX))
        
        snsPlot = sns.lineplot(x="n", y="time(s)",
             hue="Method",
             data=pdData)
        plt.yticks(RANGE, 10.0**RANGE)
        
        
    else:
        snsPlot = sns.lineplot(x="n", y="time(s)",
             hue="Method",
             data=pdData)
        
        
    h,l = snsPlot.get_legend_handles_labels()
    plt.legend(h[1:],l[1:])

    plt.ylabel("time(s)",fontsize=20)
        
        
    plt.xlabel(r"$n$",fontsize=15)

    plt.tight_layout()
    if path:
        
        fig = snsPlot.get_figure()


        fig.savefig(path)


def run_test(X,Y,bins, parallel=True, midP=False):
    if parallel:
        #Exact test
        SGM = significance_of_mean_cuda(bins, dtype_v=np.uint32,dtype_A=np.float64, verbose=False)
        SGM.run(X.reshape(1,-1),Y.reshape(1,-1), midP)
        p_val = [2 * min( p, (1-p)) for p in SGM.get_p_values()][0]
    else:
        p_val = p_value_calc([list(X), list(Y), bins])

    return p_val

def shiftMethod(X_list, y_list, bins, parallel=True, midP=False):
    pt_list = list()
    pe_list = list()
    TIME = list()

    for Xp, yp in zip(X_list, y_list):
        p_t = list()
        p_e = list()
        time_list = list()
    
        for x, y in zip(Xp, yp):
            
            
            t, p = ttest_ind(y, x)
            
       
            p_t.append(p)
        
            start = time.time()
            p_e.append(run_test(y, x, bins, parallel, midP))
            end = time.time()
        
            time_list.append(end - start)
    
        pt_list.append(p_t)
        pe_list.append(p_e)
        TIME.append(time_list)
    
    return pt_list, pe_list, TIME

def prepareBoxPandas(timeData, sampleSizes,label=None):
    preparePd = list()
    for time, sample in zip(timeData, sampleSizes):
        for t in time:
            if label:
                preparePd.append([t, sample, label])
            else:
                preparePd.append([t, sample])
    return preparePd

def SNSMultipleboxPlot(allEerrorList, Bin, allMWUList=None, allFPList=None, log=True, 
                       path=None, test_type="setSize", relError=True, dashed=False,
                       ylim=False):
    
    plt.figure(figsize=(16, 10))
    
    dataParallel = prepareBoxPandas(allEerrorList, Bin, "Parallel Green")
    
    data= dataParallel
    
    if allFPList:
        dataFastApprox = prepareBoxPandas(allFPList, Bin, "FastPerm")
        data += dataFastApprox
    if allMWUList:
        MannWhitneyApprox = prepareBoxPandas(allMWUList, Bin, "Mann–Whitney $\it{U}$ test")
        data += MannWhitneyApprox
    
    sns.set(style="white")
    sns.set_context("talk")
    
    pdData = pd.DataFrame(data, columns=['error','bins', 'Method'])
    
    if test_type=="setSize":
        snsPlot = sns.boxplot(x="bins", y="error", data=pdData, hue="Method")
    else:
        snsPlot = sns.boxplot(x="bins", y="error", data=pdData, color="skyblue")
        
    
    if relError:
        plt.ylabel(r"Relative error $\big(|\frac{p_{*}-p_{t}}{p_{t}}|)$",fontsize=25)
    else:
        plt.ylabel(r"$\frac{p_{*}}{p_{t}}$",fontsize=45)
        
    if test_type=="windowSize":
        plt.xlabel(r"$n_{w}$",fontsize=45)
    else:
        plt.xlabel(r"$n$",fontsize=45)
    
    sns.set_style("ticks")
    sns.despine()
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.30)
    
     
    
    if log:
        if allFPList and allMWUList:
            MAX = max(np.max(allEerrorList), np.max(allFPList), np.max(allMWUList))
            MIN = min(np.min(allEerrorList), np.min(allFPList), np.min(allMWUList))
        elif allFPList:
            MAX = max(np.max(allEerrorList), np.max(allFPList))
            MIN = min(np.min(allEerrorList), np.min(allFPList))
        elif allMWUList:
            MAX = max(np.max(allEerrorList), np.max(allMWUList))
            MIN = min(np.min(allEerrorList), np.min(allMWUList))
        else:
            MAX = np.max(allEerrorList)
            MIN = np.min(allEerrorList)
         

        if dashed:
            plt.axhline(0, ls=':', color="black", linewidth=3)
        
            
        #RANGE = np.arange(np.floor(MIN), np.ceil(MAX), 100)
        RANGE = np.arange(np.floor(MIN), np.ceil(MAX))
        
        if not relError:
            if ylim and ylim[1]<10:
                RANGE = np.array(sorted(list(RANGE) + [np.log10(0.5), np.log10(2)]))
            else:
                RANGE = np.array(sorted(list(RANGE)))

        plt.yticks(RANGE, 10.0**RANGE)

        
        
    if test_type=="setSize":
        plt.legend().set_title('')
    
    #€plt.yticks([0.5, 1, 2])
    
    
    if ylim:
        plt.ylim(np.log10(ylim[0]) ,np.log10(ylim[1]))
        path += "_zoom"
        
        plt.xticks(size = 32)
        plt.yticks(size = 32)
    else:
        plt.xticks(size = 32)
        plt.yticks(size = 25)
        
    

    
    if path:
        fig = snsPlot.get_figure()
        fig.savefig(path)

def getErrors(pValList, pTtestList, rel=True):
    relatvieErrorList = list()
    for pv, pt in zip(pValList, pTtestList):
        if rel:
            relatvieErrorList.append(relError(pv, pt))
        else:
            relatvieErrorList.append(frac(pv, pt))
            
    return relatvieErrorList

def relError(x,y):
    return frac((x - y), y) 
def frac(x,y):
    return x / y

def exactTest(A,B, bins=10, one_side=False):
    SGM = significance_of_mean_cuda(bins, dtype_v=np.uint32,dtype_A=np.float64, verbose=False)
    SGM.run(np.asarray(A),np.asarray(B), midP=True)
    if one_side:
        return SGM.get_p_values()
    else:
        return [2 * min( p, (1-p)) for p in SGM.get_p_values()]

def MWU(A, B, one_side=False):
    p_mw = list()
    for a,b in zip(A, B):
        if one_side:
            p_mw.append(mannwhitneyu(a,b, alternative="less")[1])
        else:
            p_mw.append(mannwhitneyu(a,b, alternative="two-sided")[1])
    return p_mw

def ttests(A,B, one_side=False):
    p_t = list()
    for x, y in zip(A, B):
        t, p = ttest_ind(y, x)
        if one_side:
            p = p/2
            if t<0:
                p = 1-p
        p_t.append(p)
    return p_t

def multiple_plot(df, save_name):
    sns.set(style="white")
    sns.set_context("talk")
    low = min(df["Theoretical p-value"])/2
    hi = max(df["Theoretical p-value"])
    g=sns.lmplot(x='Theoretical p-value', y ='Observed p-value', data=df, 
                  fit_reg=False, height=7, truncate=True, scatter_kws={"s": 15}, hue="Test")
    
    g.set(xscale="log", yscale="log")
    g._legend.remove()


    plt.xlabel(r'Theoretical $\it{p}$ value', fontsize=24)
    plt.ylabel(r'Observed $\it{p}$ value', fontsize=24)
    
    axes = g.axes
    g.set(ylim=(low,hi), xlim=(low,hi))
    plt.plot([low,hi],[low,hi], "k", linewidth=1)
    plt.plot([2 * low,2 * hi],[low,hi], "--k", linewidth=1)
    plt.plot([low / 2,hi / 2],[low,hi], "--k", linewidth=1)
    sns.set_style("ticks")
    sns.despine()
    g.fig.tight_layout()
    
    
    plt.legend().set_title('')
    plt.legend(loc=2,prop={'size': 24})
    plt.xticks(size = 24)
    plt.yticks(size = 24)


    
    g.savefig(save_name)


def getdf(P, num_examples, test=None):
    P.sort()
    p_arr = np.array(P)
    offset = 1.0/float(num_examples)
    ideal_arr = np.linspace(offset,1.0-offset,num_examples)
    if test:
        Pdf = pd.DataFrame({'Observed p-value':p_arr,'Theoretical p-value':ideal_arr, "Test":[test]*ideal_arr.shape[0]})
    return Pdf

def memoryPlotSNS(df,binVar=False, log=False, path=None):
    fig, ax = plt.subplots()

    sns.set(style="white")
    sns.set_context("talk")
    
    
    
    pdData = df
    
    palette = dict(zip(set(df.Experiment), ["r", "g", "b"]))
    
    x = pdData["time(s)"].values
    
    if log:        
        MAX = max(x)
        MIN = min(x)
        


        RANGE = np.arange(np.floor(MIN), np.ceil(MAX))
        snsPlot = sns.lineplot(x="Sample size", y="time(s)",
             hue="Experiment",
             data=pdData, palette=palette)#.set(yticks = RANGE, yticklabels=10**RANGE)
        plt.yticks(RANGE, 10.0**RANGE)
        
        
    else:
        snsPlot = sns.lineplot(x="Sample size", y="time(s)",
             hue="Experiment",
             data=pdData,palette=palette)
    
    if binVar:
        plt.xlabel(r"$n$")
        
    else:
        plt.xlabel(r"$n$")
    
    
    
    plt.legend(loc='upper left')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    

    plt.setp(snsPlot.get_legend().get_texts(), fontsize='12')
    
    sns.despine()
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.30)


def bootstrap(invec):
    """ Function for generating bootstrap sampled versions of a vector """
    idx = npr.randint(0, len(invec), len(invec))
    return [invec[i] for i in idx]

def estimatePi0(p, numBoot=100, numLambda=100, maxLambda=0.95):
    """
    Function for estimaring pi_0, i.e. the prior null probability
    for p values.
    Args:
        p (list(float)): The list of p values for which pi_0 should be estimated
        numBoot (int): The number of bootstrap rounds that should be made.
        numLambda (int): The number of lambda tresholds that should be evaluated.
        maxLambda (float): The upper bond of the range of lambda treshold.
    Returns:
        pi_0, a float containing the pi_0 estimate.
    """
    p.sort()
    n=len(p)
    lambdas=np.linspace(maxLambda/numLambda,maxLambda,numLambda)
    Wls=np.array([n-np.argmax(p>=l) for l in lambdas])
    pi0s=np.array([Wls[i] / (n * (1 - lambdas[i])) for i in range(numLambda)])
    minPi0=np.min(pi0s)
    mse = np.zeros(numLambda)
    for boot in range(numBoot):
        pBoot = bootstrap(p)
        pBoot.sort()
        WlsBoot =np.array([n-np.argmax(pBoot>=l) for l in lambdas])
        pi0sBoot =np.array([WlsBoot[i] / (n *(1 - lambdas[i])) for i in range(numLambda)])
        mse = mse + np.square(pi0sBoot-minPi0)
    minIx = np.argmin(mse)
    return pi0s[minIx]

def qvalues(pvalues,p_col = "p", q_col ="q", pi0 = 1.0):
    """
    Function for estimaring q values.
    Args:
        pvalues (DataFrame): A DataFrame that contain at least one column with pvalues.
        p_col (str): The name of the column that contain pvalues.
        q_col (str): The name of the column that that shall contain the estimated
                    q-values. The column will be created if not already existing.
        pi0 (float): The prior probability of the null hypothesis. If set to None, this is estimated from data.
                    Defaults to 1.0
    Returns:
        The modified DataFrame.
    """
    m = pvalues.shape[0] # The number of p-values
    pvalues.sort_values(p_col,inplace=True) # sort the pvalues in acending order
    if pi0 is None:
        pi0 = estimatePi0(list(pvalues[p_col].values))

    # calculate a FDR(t) as in Storey & Tibshirani
    num_p = 0.0
    for ix in pvalues.index:
        num_p += 1.0
        fdr = pi0*pvalues.loc[ix,p_col]*m/num_p
        pvalues.loc[ix,q_col] = fdr

    # calculate a q(p) as the minimal FDR(t)
    old_q=1.0
    for ix in reversed(list(pvalues.index)):
        q = min(old_q,pvalues.loc[ix,q_col])
        old_q = q
        pvalues.loc[ix,q_col] = q
    return pvalues

def timePlotSNSFastpermCoinFastPerm(TIMEParallel, TIME_MC, TIMEsingleThread, TimeCoin, sampleShape, log=False, path=None):
    
    sns.set(style="white")
    sns.set_context("talk")
    
    preparePdParallel = preparePandasFastperm(TIMEParallel, sampleShape, "Parallel Green")
    preparePdMc = preparePandasFastperm(TIME_MC, sampleShape, 'FastPerm')
    preparePdSingle = preparePandasFastperm(TIMEsingleThread, sampleShape, 'Single thread Green')
    preparePdCoin = preparePandasFastperm(TimeCoin, sampleShape, 'Coin')
    
    data = preparePdMc + preparePdParallel + preparePdSingle + preparePdCoin
    
    
        
    pdData = pd.DataFrame(data, columns=['Method', 'time(s)','n'])

    
    if log:        
        MAX = max(np.max(TIMEParallel), np.max(TIME_MC), np.max(TIMEsingleThread), np.max(TimeCoin))
        MIN = min(np.min(TIMEParallel), np.min(TIME_MC), np.min(TIMEsingleThread), np.min(TimeCoin))
            
            
        
        RANGE = np.arange(np.floor(MIN), np.ceil(MAX))
        
        snsPlot = sns.lineplot(x="n", y="time(s)",
             hue="Method",
             data=pdData)
        plt.yticks(RANGE, 10.0**RANGE)
        
        
    else:
        snsPlot = sns.lineplot(x="n", y="time(s)",
             hue="Method",
             data=pdData)
        
        
    h,l = snsPlot.get_legend_handles_labels()
    plt.legend(h[1:],l[1:])

    plt.ylabel("time(s)",fontsize=20)
        
        
    plt.xlabel(r"$n$",fontsize=15)

    plt.tight_layout()
    if path:
        
        fig = snsPlot.get_figure()


        fig.savefig(path)

def plotSNS(tup1, tup2, tup3 , sampleShape, y_lab, x_lab, log=False, path=None):
    
    dataList1, name1 = tup1 
    dataList2, name2 = tup2 
    dataList3, name3 = tup3

    
    sns.set(style="white")
    sns.set_context("talk")
    
    df1 = preparePandasFastperm(dataList1, sampleShape, name1)
    df2 = preparePandasFastperm(dataList2, sampleShape, name2)
    df3 = preparePandasFastperm(dataList3, sampleShape, name3)

    
    data = df1 + df2 + df3
    
    
        
    pdData = pd.DataFrame(data, columns=['Method', 'time(s)','n'])

    
    if log:        
        MAX = max(np.max(dataList1), np.max(dataList2), np.max(dataList3))
        MIN = min(np.min(dataList1), np.min(dataList2), np.min(dataList3))
            
            
        
        RANGE = np.arange(np.floor(MIN), np.ceil(MAX))
        
        snsPlot = sns.lineplot(x="n", y="time(s)",
             hue="Method",
             data=pdData)
        plt.yticks(RANGE, 10.0**RANGE)
        
        
    else:
        snsPlot = sns.lineplot(x="n", y="time(s)",
             hue="Method",
             data=pdData)
        
        
    h,l = snsPlot.get_legend_handles_labels()
    plt.legend(h[1:],l[1:])

    plt.ylabel(y_lab,fontsize=20)
        
        
    plt.xlabel(x_lab,fontsize=15)

    plt.tight_layout()
    if path:
        
        fig = snsPlot.get_figure()


        fig.savefig(path)


    





    


