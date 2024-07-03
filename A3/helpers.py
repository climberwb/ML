import numpy as np
import random
import torch


def generate_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
def seed_decorator(seed=42):
    def decorator(func):
        def wrapper(*args, **kwargs):
            generate_seed(seed)
            return func(*args, **kwargs)
        return wrapper
    return decorator



def increase_font_size(plt, label_fontsize=14, title_fontsize=16, legend_fontsize=12*1.3, tick_fontsize=12, contant=1.3):
    """
    Increases the font size for labels, title, legend, and tick labels of the current plot.
    
    Parameters:
    plt (module): The Matplotlib pyplot module.
    label_fontsize (int): Font size for the x and y labels.
    title_fontsize (int): Font size for the title.
    legend_fontsize (int): Font size for the legend.
    tick_fontsize (int): Font size for the tick labels.
    """
    if contant:
        label_fontsize = int(label_fontsize * contant)
        title_fontsize = int(title_fontsize * contant)
        legend_fontsize = int(legend_fontsize * contant)
        tick_fontsize = int(tick_fontsize * contant)
        
    ax = plt.gca()
    ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
    ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)
    ax.set_title(ax.get_title(), fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontsize(legend_fontsize)
        legend.set_title(legend.get_title().get_text(), prop={'size': legend_fontsize})
        
        
        
        
def increase_font_size_subplots_included(plt, label_fontsize=14, title_fontsize=16, legend_fontsize=12*1.3, tick_fontsize=12, constant=1.3):
    """
    Increases the font size for labels, title, legend, and tick labels of the current plot and subplots.
    
    Parameters:
    plt (module): The Matplotlib pyplot module.
    label_fontsize (int): Font size for the x and y labels.
    title_fontsize (int): Font size for the title.
    legend_fontsize (int): Font size for the legend.
    tick_fontsize (int): Font size for the tick labels.
    """
    if constant:
        label_fontsize = int(label_fontsize * constant)
        title_fontsize = int(title_fontsize * constant)
        legend_fontsize = int(legend_fontsize * constant)
        tick_fontsize = int(tick_fontsize * constant)
    
    # Get the current figure and its axes
    fig = plt.gcf()
    axes = fig.get_axes()
    
    for ax in axes:
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)
        ax.set_title(ax.get_title(), fontsize=title_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(legend_fontsize)
            legend.set_title(legend.get_title().get_text(), prop={'size': legend_fontsize})
