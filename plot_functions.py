#!/usr/bin/env python
###########################################################################################
#
# 			Class for plotting results of Cifar10 analysis.
#
###########################################################################################

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

###########################################################################################


def plot_images(images, cls_true, class_names, smooth=True):
    """
	Plots passed images on the 10 by 10 grid.   
    
    """
    assert len(images) == len(cls_true) == 100

    # Create figure with sub-plots.
    fig, axes = plt.subplots(10, 10,figsize = (12,12))
    
    # Adjust vertical spacing
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
         
    # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Label each image row
    for ax, row_class in zip(axes[:,0], class_names):
        ax.set_ylabel(row_class, rotation=0, size='large')
        ax.yaxis.labelpad = 35
    plt.show()



def plot_confusion_matrix(cm, normalize=False, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix.    
    If normalize is set to True, the rows of the confusion matrix are normalized so that they sum up to 1.
    
    """
    if normalize is True:
        cm = cm/cm.sum(axis=1)[:, np.newaxis]
        vmin, vmax = 0., 1.
        fmt = '.2f'
    else:
        vmin, vmax = None, None
        fmt = 'd'
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=vmin, vmax=vmax, 
                    annot=True, annot_kws={"fontsize":12}, fmt=fmt)
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def scatter_2d_label(X_2d, y,labels, s=2, alpha=0.5, lw=2):
    """Visualuse a 2D embedding with corresponding labels.
    
    X_2d : ndarray, shape (n_samples,2)
        Low-dimensional feature representation.
    
    y : ndarray, shape (n_samples,)
        Labels corresponding to the entries in X_2d.
        
    s : float
        Marker size for scatter plot.
    
    alpha : float
        Transparency for scatter plot.
        
    lw : float
        Linewidth for scatter plot.
    """
    targets = np.unique(y)
    colors = sns.color_palette(n_colors=targets.size)
    for color, target in zip(colors, targets):
        plt.scatter(X_2d[y == target, 0], X_2d[y == target, 1], color=color, label=labels[int(target)], s=s, alpha=alpha, lw=lw)


