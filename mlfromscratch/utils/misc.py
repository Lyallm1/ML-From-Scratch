import progressbar, matplotlib.pyplot as plt, numpy as np
from mlfromscratch.utils.data_operation import calculate_covariance_matrix

bar_widgets = ['Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"), ' ', progressbar.ETA()]

class Plot():
    def __init__(self): 
        self.cmap = plt.get_cmap('viridis')

    def _transform(self, X, dim):
        eigenvalues, eigenvectors = np.linalg.eig(calculate_covariance_matrix(X))
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:dim]
        return X.dot(np.atleast_1d(eigenvectors[:, idx])[:, :dim])

    def plot_regression(self, lines, title, axis_labels=None, mse=None, scatter=None, legend={"type": "lines", "loc": "lower right"}):
        if scatter:
            scatter_plots = scatter_labels = []
            for s in scatter:
                scatter_plots += [plt.scatter(s["x"], s["y"], color=s["color"], s=s["size"])]
                scatter_labels += [s["label"]]
            scatter_plots = tuple(scatter_plots)
            scatter_labels = tuple(scatter_labels)
        if mse:
            plt.suptitle(title)
            plt.title("MSE: %.2f" % mse, fontsize=10)
        else: plt.title(title)
        if axis_labels:
            plt.xlabel(axis_labels["x"])
            plt.ylabel(axis_labels["y"])
        plt.legend(loc="lower_left") if legend["type"] == "lines" or (legend["type"] == "scatter" and scatter) else plt.legend(scatter_plots, scatter_labels, loc=legend["loc"])
        plt.show()

    def plot_in_2d(self, X, y=None, title=None, accuracy=None, legend_labels=None):
        x1, x2 = self._transform(X, dim=2)[:]
        class_distr = []
        y = np.array(y).astype(int)
        colors = [self.cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]
        for i, l in enumerate(np.unique(y)): class_distr.append(plt.scatter(x1[y == l], x2[y == l], color=colors[i]))
        if not legend_labels is None: plt.legend(class_distr, legend_labels, loc=1)
        if title:
            if accuracy:
                plt.suptitle(title)
                plt.title("Accuracy: %.1f%%" % (100 * accuracy), fontsize=10)
            else: plt.title(title)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    def plot_in_3d(self, X, y=None):
        x1, x2, x3 = self._transform(X, dim=3)[:]
        plt.figure().add_subplot(111, projection='3d').scatter(x1, x2, x3, c=y)
        plt.show()
