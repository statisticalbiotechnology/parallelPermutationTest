import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def preparePandas(timeData, sampleSizes, name):
    """Create DF for sns-plots"""
    preparePd = list()
    for time, sample in zip(timeData, sampleSizes):
        preparePd.append([str(name),time, sample])
        
            
    return preparePd

def timePlotSNS(TimeList_list, NameList, sampleShape, y_label , x_label ,log=False, path=None):
    """Create SNS timeseries-plot"""
    a4_dims = (11.7/1.5, 8.27/1.5)
    fig, ax = plt.subplots(figsize=a4_dims)
    

    
    sns.set(style="white")
    sns.set_context("talk")
    
    for i, (time_list, name) in enumerate(zip(TimeList_list, NameList)):
        if log:
            df = preparePandas(np.log10(time_list), sampleShape, name)
        else:
            df = preparePandas(time_list, sampleShape, name)
            
        if i ==0:
            data = df
        else:
            data += df
            
    pdData = pd.DataFrame(data, columns=['Method', y_label,'bins'])
    

    if log:        
        MAX = max(pdData[y_label])
        MIN = min(pdData[y_label])

        RANGE = np.arange(np.floor(MIN), np.ceil(MAX))
        snsPlot = sns.lineplot(x="bins", y=y_label,
             hue="Method",
             data=pdData)#.set(yticks = RANGE, yticklabels=10**RANGE)
        plt.yticks(RANGE, 10.0**RANGE)
        
        
    else:
        snsPlot = sns.lineplot(x="bins", y=y_label,
             hue="Method",
             data=pdData,)
    

    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    
    
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    
    plt.setp(snsPlot.get_legend().get_texts(), fontsize=20)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.tight_layout()
    
    
    
    if path:   
        fig = snsPlot.get_figure()
        fig.savefig(path)

def memoryPlotSNS(ALLTimeList, names, variables, binVar=False, log=False, path=None, colors= ["r", "g", "b"]):
    """ Create memory plit"""
    
    def getXrangeData(setSize, maxRange, batchS):
        """Get range of data"""
        
        AN, BN = [np.random.normal(0,1,setSize) for i in range(maxRange)], [np.random.normal(0,1,setSize) for i in range(maxRange)]
        
        batchList = list()
        
        for i in range(0, len(AN), batchS):
            
            Abatch = AN[:i+batchS]
            batchList.append(len(Abatch))

        return batchList
    
    def getScatterData(sN,tList, maxRange, batchSize):
        y=list()
        x=list()
        for i, j in enumerate(range(0, maxRange, batchSize)):
            if j % sN ==0 and j!=0:
                y.append(j)
                x.append(tList[i-1])
        return x, y
    
    
    a4_dims = (11.7/1.5, 8.27/1.5)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.set(style="white")
    sns.set_context("talk")
    
    
    setN, sampleNList, sampleRangeMax, batchsize = variables
    batchList = getXrangeData(setN, sampleRangeMax, batchsize)
    
    
    for i, (time_list, name) in enumerate(zip(ALLTimeList, names)):
        if log:
            df = preparePandas(np.log10(time_list), batchList, name)
        else:
            df = preparePandas(time_list, batchList, name)
            
        if i ==0:
            data = df
        else:
            data += df
            
    df = pd.DataFrame(data, columns=['Experiment', 'time(s)',"Sample size"])
    
    palette = dict(zip(set(df.Experiment), colors))
    
    x = df["time(s)"].values
    
    if log:        
        MAX = max(x)
        MIN = min(x)
        


        RANGE = np.arange(np.floor(MIN), np.ceil(MAX))
        snsPlot = sns.lineplot(x="Sample size", y="time(s)",
             hue="Experiment",
             data=df, palette=palette)#.set(yticks = RANGE, yticklabels=10**RANGE)
        plt.yticks(RANGE, 10.0**RANGE)
        
        
    else:
        snsPlot = sns.lineplot(x="Sample size", y="time(s)",
             hue="Experiment",
             data=df,palette=palette)
    
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
    
    for N, Time, n in zip(sampleNList, ALLTimeList, names):
        x, y = getScatterData(N,Time, sampleRangeMax, batchsize)
        plt.scatter(y, x, marker="o", color=palette[n])