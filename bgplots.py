import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

# Print graphics
def plot_results(results, param_values):
    x = [1,2]   # X axis
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin = 0, vmax = (len(param_values) - 1))
    scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)

    fig, ax = plt.subplots()

    for i in range(len(param_values)):
        p = param_values[i]
        colorVal = scalarMap.to_rgba(i)
        ax.plot(x, results[p], "o-", color=colorVal)
    
    ax.axis([0.75, 2.25, 0.48, 0.95])
    ax.set_ylabel("Mean Accuracy")
    ax.set_xticks([1,2])
    ax.set_xticklabels(["Choose", "Avoid"])
    #fig.colorbar(ax, param_values)
    plt.legend(["T = %.2f" % x for x in param_values], loc="best")
    plt.show()

def plot_rocs(result_list, color_list, legend=None, title=None):
    plt.axis([-0.025, 0.30, 0.5, 0.85])
    plt.ylabel("Mean accuracy")
    plt.xlabel("Estimate bias")
    plt.title("Accuracy vs. Bias")
    
    tops = []
    
    for results, color in zip(result_list, color_list):
        params = sorted(results.keys())
        accuracies = [results[x] for x in params]
        biases = [x[0] - x[1] for x in accuracies]
        means = [np.mean(x) for x in accuracies]
        tops.append(np.max(means))
        plt.plot(biases, means, "o-", color=color)
        
    # Plot the max theoretical
    for j in range(len(color_list)):
        top = tops[j]
        color = color_list[j]
        plt.plot([-0.025, 0.30], [top, top], "--", color = color)
        
    plt.plot([0, 0], [0.5, 1], "--", color="grey")
    if legend is not None:
        plt.legend(legend, loc="best")
    plt.grid()
    if title is not None:
        plt.title(title)
        
    plt.show()

def plot_ca(results, color_list, title=None):
    T = sorted(results.keys())
    choose = [results[x][0] for x in T]
    avoid = [results[x][1] for x in T]
    plt.plot(T, choose, "o-", color=color_list[0])
    plt.plot(T, avoid, "o-", color=color_list[1])
    plt.axis([-0.1, 4.1, 0.5, 0.92])
    plt.xlabel(r"Temperature $T$")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend(["Choose accuracy", "Avoid accuracy"], loc="best")
    if title is not None:
        plt.title(title)
    plt.show()
