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

def timePlotSNS(TimeList_list, NameList, sampleShape, y_label , x_label ,log=False, path=None,):
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

    if path:   
        fig = snsPlot.get_figure()
        fig.savefig(path)

def SNSMultipleboxPlot(data, name, Bin, log=True, 
                       path=None, test_type="setSize", relError=True, dashed=False,
                       ylim=False):
    
    def prepareBoxPandas(timeData, sampleSizes,label=None):
        preparePd = list()
        for time, sample in zip(timeData, sampleSizes):
            for t in time:
                if label:
                    preparePd.append([t, sample, label])
                else:
                    preparePd.append([t, sample])
        return preparePd
    
    a4_dims = (11.7 * 2 , 8.27 * 1.5)
    fig, ax = plt.subplots(figsize=a4_dims)
    
    sns.set(style="white")
    sns.set_context("talk")
    
    for i,(d, n) in enumerate(zip(data, name)):
        if log:
            df = prepareBoxPandas(np.log10(d), Bin, n)
        else:
            df = preparePandas(d, Bin, n)
            
        if i ==0:
            data = df
        else:
            data += df
        
    
    pdData = pd.DataFrame(data, columns=['error','bins', 'Method'])
    

    
    if test_type=="setSize":
        snsPlot = sns.boxplot(x="bins", y="error", data=pdData, hue="Method")
    else:
        snsPlot = sns.boxplot(x="bins", y="error", data=pdData, color=sns.xkcd_rgb["medium green"])
        
    
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
        MAX = max(pdData["error"])
        MIN = min(pdData["error"])
        

        if dashed:
            plt.axhline(0, ls=':', color="black", linewidth=3)
        
            
        RANGE = np.arange(np.floor(MIN), np.ceil(MAX))
        
        if not relError:
            if ylim and ylim[1]<10:
                RANGE = np.array(sorted(list(RANGE) + [np.log10(0.5), np.log10(2)]))
            else:
                RANGE = np.array(sorted(list(RANGE)))

        plt.yticks(RANGE, 10.0**RANGE)

    

        
    if test_type=="setSize":
        plt.legend().set_title('')
        plt.setp(ax.get_legend().get_texts(), fontsize='32') # for legend text
        
    

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

def timePlotSNSMC(TimeList_list, NameList, sampleShape, y_label , x_label , palette, log=False, path=None):
    """Create SNS timeseries-plot"""
    a4_dims = (11.7, 8.27)
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
             data=pdData,palette=palette)#.set(yticks = RANGE, yticklabels=10**RANGE)
        plt.yticks(RANGE, 10.0**RANGE)
        
        
    else:
        snsPlot = sns.lineplot(x="bins", y=y_label,
             hue="Method",
             data=pdData,palette=palette)
    

    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    
    
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    
    plt.setp(snsPlot.get_legend().get_texts(), fontsize=20)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.tight_layout()
    
    ax.lines[3].set_linestyle("--")
    ax.lines[4].set_linestyle("--")
    ax.lines[5].set_linestyle("--")
    
    leg = ax.legend()
    leg_lines = leg.get_lines()
    leg_lines[4].set_linestyle("--")
    leg_lines[5].set_linestyle("--")
    leg_lines[6].set_linestyle("--")

    
    if path:   
        fig = snsPlot.get_figure()
        fig.savefig(path)


def log_vs_log_plot(TimeList_list, NameList, sampleShape, y_label , x_label ,log=False, path=None,):
    """Create SNS timeseries-plot"""
    #a4_dims = (11.7*2, 8.27)
    #fig, ax = plt.subplots(figsize=a4_dims)
    

    
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
        g = sns.lmplot(x="bins", y=y_label,
             hue="Method",
             data=pdData,scatter_kws={"s": 15}, fit_reg=False, height=7)
    
    g._legend.remove()
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    
    
    axes = g.axes
    
    sns.set_style("ticks")
    sns.despine()
    g.fig.tight_layout()
    
    
    plt.legend().set_title('')
    plt.legend(loc=2,prop={'size': 24})
    plt.xticks(size = 24)
    plt.yticks(size = 24)

    
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.tight_layout()
    
    mul = 5
    low = min(sampleShape)/2
    hi = max(sampleShape)
    
    plt.plot([low,hi],[low,hi], "k", linewidth=1)
   
    
    if path:   
        g.savefig(path)

