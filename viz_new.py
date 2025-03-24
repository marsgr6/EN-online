import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import ipywidgets as widgets
from ipywidgets import interact
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def plot_data(all_data):
    """
    Interactive data visualization tool for multiple datasets and plot types.
    
    Parameters:
    - all_data (dict): Dictionary of pandas DataFrames with dataset names as keys.
    """
    
    # Helper function to filter columns by dtype
    def get_filtered_columns(data, dtype="all"):
        """Filter columns by dtype: 'object', 'number', or 'all'."""
        if dtype == "object":
            return data.select_dtypes(include="object").columns
        elif dtype == "number":
            return data.select_dtypes(include=np.number).columns
        return data.columns

    # Plotting functions defined first
    def plot_correlation(data, **kwargs):
        data_c = data.drop(data.columns[data.nunique() == 1], axis=1)
        corr_matrix = data_c.select_dtypes(include=np.number).corr()
        sns.clustermap(corr_matrix, annot=True, cmap="coolwarm")
        plt.show()

    def plot_clustermap(data, z_score=None, standard_scale=None):
        data_c = data.drop(data.columns[data.nunique() == 1], axis=1).select_dtypes(include="number")
        sns.clustermap(data_c.dropna(), annot=True, z_score=z_score, standard_scale=standard_scale, cmap="coolwarm")
        plt.show()

    def plot_pairplot(data, hue_var, kind="kde"):
        sns.pairplot(data, hue=hue_var, diag_kind=kind)
        plt.show()

    def plot_missingno(data, tplot="matrix"):
        if tplot == "matrix":
            msno.matrix(data)
        elif tplot == "bars":
            msno.bar(data)
        elif tplot == "heatmap":
            msno.heatmap(data)
        elif tplot == "dendrogram":
            msno.dendrogram(data)
        plt.show()

    def plot_bars(data, var_x, hue, tplot="bars"):
        plt.figure(figsize=(8, 6))
        if tplot == "bars":
            ax = sns.countplot(data=data, x=var_x, hue=hue)
            for container in ax.containers:
                ax.bar_label(container)
        else:
            df_2dhist = pd.DataFrame({x_label: grp[var_x].value_counts() for x_label, grp in data.groupby(hue)})
            sns.heatmap(df_2dhist, cmap="viridis", annot=True, cbar=False, fmt=".0f")
            plt.xlabel(hue)
            plt.ylabel(var_x)
        plt.show()

    def plot_boxes(data, var_x, var_y, hue, tplot="boxplot", no_hue=False):
        palette = sns.color_palette("Set2", 12)
        plt.figure(figsize=(8, 6))
        hue_to_use = None if no_hue else hue
        if tplot == "boxplot":
            sns.boxplot(data=data, x=var_x, y=var_y, hue=hue_to_use, palette=palette)
        elif tplot == "violin":
            if len(np.unique(data[hue])) == 2 and not no_hue:
                sns.violinplot(data=data, x=var_x, y=var_y, hue=hue, split=True, cut=0, palette=palette, inner="quartile")
            else:
                sns.violinplot(data=data, x=var_x, y=var_y, hue=hue_to_use, cut=0, palette=palette, inner="quartile")
        elif tplot == "lineplot":
            sns.lineplot(data=data, x=var_x, y=var_y, hue=hue_to_use, err_style="bars", errorbar=("ci", 68), estimator="mean", palette=palette)
        plt.show()

    def plot_ridges(data, var_x, var_y, hue_var, no_hue=False):
        hue_to_use = var_x if no_hue else hue_var
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), "axes.linewidth": 2})
        palette = sns.color_palette("Set2", 12)
        g = sns.FacetGrid(data, palette=palette, row=var_x, hue=hue_to_use, aspect=6, height=1.5)
        g.map_dataframe(sns.kdeplot, x=var_y, cut=0, fill=True, alpha=0.5)
        g.add_legend()
        plt.show()

    def plot_histogram(data, var_x, hue_var, multiple="layer", stat="count", element="bars", common_norm=False, cumulative=False):
        plt.figure(figsize=(8, 6))
        palette = sns.color_palette("Set2", 12)
        sns.histplot(data=data, x=var_x, hue=hue_var, stat=stat, element=element, multiple=multiple, 
                     common_norm=common_norm, cumulative=cumulative, palette=palette)
        plt.show()

    def plot_density1(data, var_x, hue_var, multiple="layer", common_norm=False, cumulative=False):
        plt.figure(figsize=(8, 6))
        palette = sns.color_palette("Set2", 12)
        sns.kdeplot(data=data, x=var_x, hue=hue_var, multiple=multiple, cut=0, fill=True, 
                    common_norm=common_norm, cumulative=cumulative, palette=palette)
        plt.show()

    def plot_density2(data, var_x, var_y, hue_var, col_var, kind="kde", common_norm=False, cumulative=False, facet=False, rug=False):
        plt.figure(figsize=(8, 6))
        palette = sns.color_palette("Set2", 12)
        if facet:
            sns.displot(data=data, x=var_x, y=var_y if var_x != var_y else None, hue=hue_var, col=col_var, kind=kind, 
                        rug=rug, common_norm=common_norm, cumulative=cumulative, palette=palette)
        else:
            sns.displot(data=data, x=var_x, y=var_y if var_x != var_y else None, hue=hue_var, kind=kind, 
                        rug=rug, common_norm=common_norm, cumulative=cumulative, palette=palette)
        plt.show()

    def plot_scatter(data, var_x, var_y, hue, style, size, alpha=0.5, use_style=False):
        plt.figure(figsize=(8, 6))
        if use_style:
            sns.scatterplot(data=data, x=var_x, y=var_y, hue=hue, style=style, size=size, alpha=alpha)
        else:
            sns.scatterplot(data=data, x=var_x, y=var_y, hue=hue, size=size, alpha=alpha)
        plt.show()

    def plot_catplot(data, var_x, var_y, hue, col, kind="strip", facet=False):
        plt.figure(figsize=(8, 6))
        if facet:
            sns.catplot(data=data, x=var_x, y=var_y, hue=hue, col=col, kind=kind)
        else:
            sns.catplot(data=data, x=var_x, y=var_y, hue=hue, kind=kind)
        plt.show()

    def plot_regression(data, var_x, var_y, hue_var, order=1, ci=95, use_hue=True):
        plt.figure(figsize=(8, 6))
        if use_hue:
            sns.lmplot(data=data, x=var_x, y=var_y, hue=hue_var, order=order, ci=ci)
        else:
            sns.lmplot(data=data, x=var_x, y=var_y, order=order, ci=ci)
        plt.show()

    # Define plot type configurations after functions
    PLOT_TYPES = {
        "bars": {"func": plot_bars, "interactive": True},
        "boxes": {"func": plot_boxes, "interactive": True},
        "ridges": {"func": plot_ridges, "interactive": True},
        "histogram": {"func": plot_histogram, "interactive": True},
        "density 1": {"func": plot_density1, "interactive": True},
        "density 2": {"func": plot_density2, "interactive": True},
        "scatter": {"func": plot_scatter, "interactive": True},
        "catplot": {"func": plot_catplot, "interactive": True},
        "regression": {"func": plot_regression, "interactive": True},
        "correlation": {"func": plot_correlation, "interactive": False},
        "clustermap": {"func": plot_clustermap, "interactive": True},
        "pairplot": {"func": plot_pairplot, "interactive": True},
        "missingno": {"func": plot_missingno, "interactive": True},
    }

    @interact(
        ds=list(all_data.keys()),
        plot_type=list(PLOT_TYPES.keys()),
        risk_it_all=False
    )
    def main_interact(ds, plot_type, risk_it_all):
        data = all_data[ds].copy()
        
        # Convert categorical columns to strings
        for col in data.select_dtypes(include="category").columns:
            data[col] = data[col].astype("str")

        # Adjust column types based on risk_it_all
        col_types = {
            "var_x": "all" if risk_it_all else "object",
            "var_y": "all" if risk_it_all else "number",
            "hue": "all" if risk_it_all else "object",
            "style": "all" if risk_it_all else "object",
            "size": "all" if risk_it_all else "number",
            "col": "all" if risk_it_all else "object",
            "hue_var": "all" if risk_it_all else "object",
            "col_var": "all" if risk_it_all else "object"
        }

        # Get plot configuration
        plot_config = PLOT_TYPES[plot_type]
        plot_func = plot_config["func"]

        if not plot_config["interactive"]:
            plot_func(data)
        else:
            # Dynamically create widget arguments based on plot type
            if plot_type == "clustermap":
                interact(plot_func, data=widgets.fixed(data), z_score=[None, 0, 1], standard_scale=[None, 0, 1])
            elif plot_type == "pairplot":
                interact(plot_func, data=widgets.fixed(data), hue_var=get_filtered_columns(data, "object"), kind=["kde", "hist"])
            elif plot_type == "missingno":
                interact(plot_func, data=widgets.fixed(data), tplot=["matrix", "bars", "heatmap", "dendrogram"])
            elif plot_type == "bars":
                interact(plot_func, data=widgets.fixed(data), var_x=get_filtered_columns(data, col_types["var_x"]), 
                         hue=get_filtered_columns(data, col_types["hue"]), tplot=["bars", "heatmap"])
            elif plot_type == "boxes":
                interact(plot_func, data=widgets.fixed(data), var_x=get_filtered_columns(data, col_types["var_x"]), 
                         var_y=get_filtered_columns(data, col_types["var_y"]), hue=get_filtered_columns(data, col_types["hue"]), 
                         tplot=["boxplot", "lineplot", "violin"], no_hue=False)
            elif plot_type == "ridges":
                interact(plot_func, data=widgets.fixed(data), var_x=get_filtered_columns(data, col_types["var_x"]), 
                         var_y=get_filtered_columns(data, col_types["var_y"]), hue_var=get_filtered_columns(data, col_types["hue_var"]), 
                         no_hue=False)
            elif plot_type == "histogram":
                interact(plot_func, data=widgets.fixed(data), var_x=get_filtered_columns(data, col_types["var_x"]), 
                         hue_var=get_filtered_columns(data, col_types["hue_var"]), multiple=["layer", "dodge", "stack", "fill"], 
                         stat=["count", "probability", "percent", "density"], element=["bars", "step", "poly"], 
                         common_norm=False, cumulative=False)
            elif plot_type == "density 1":
                interact(plot_func, data=widgets.fixed(data), var_x=get_filtered_columns(data, col_types["var_x"]), 
                         hue_var=get_filtered_columns(data, col_types["hue_var"]), multiple=["layer", "stack", "fill"], 
                         common_norm=False, cumulative=False)
            elif plot_type == "density 2":
                interact(plot_func, data=widgets.fixed(data), var_x=get_filtered_columns(data, col_types["var_x"]), 
                         var_y=get_filtered_columns(data, col_types["var_y"]), hue_var=get_filtered_columns(data, col_types["hue_var"]), 
                         col_var=get_filtered_columns(data, col_types["col_var"]), kind=["hist", "kde", "ecdf"], 
                         common_norm=False, cumulative=False, facet=False, rug=False)
            elif plot_type == "scatter":
                interact(plot_func, data=widgets.fixed(data), var_x=get_filtered_columns(data, col_types["var_x"]), 
                         var_y=get_filtered_columns(data, col_types["var_y"]), hue=get_filtered_columns(data, col_types["hue"]), 
                         style=get_filtered_columns(data, col_types["style"]), size=get_filtered_columns(data, col_types["size"]), 
                         alpha=widgets.FloatSlider(value=0.5, min=0, max=1, step=0.01), use_style=False)
            elif plot_type == "catplot":
                interact(plot_func, data=widgets.fixed(data), var_x=get_filtered_columns(data, col_types["var_x"]), 
                         var_y=get_filtered_columns(data, col_types["var_y"]), hue=get_filtered_columns(data, col_types["hue"]), 
                         col=get_filtered_columns(data, col_types["col"]), kind=["strip", "swarm"], facet=False)
            elif plot_type == "regression":
                interact(plot_func, data=widgets.fixed(data), var_x=get_filtered_columns(data, col_types["var_x"]), 
                         var_y=get_filtered_columns(data, col_types["var_y"]), hue_var=get_filtered_columns(data, col_types["hue_var"]), 
                         order=[1, 2, 3], ci=[68, 95, 99, 0], use_hue=True)

# Example usage
if __name__ == "__main__":
    # Sample data for testing
    sample_data = {
        "dataset1": pd.DataFrame({
            "A": np.random.randn(100),
            "B": np.random.randn(100),
            "C": pd.Categorical(["x", "y", "z"] * 33 + ["x"]),
            "D": np.random.choice(["cat", "dog", "bird"], 100)
        }),
        "dataset2": pd.DataFrame({
            "X": np.random.randn(50),
            "Y": np.random.randint(0, 10, 50),
            "Z": pd.Categorical(["low", "high"] * 25)
        })
    }
    plot_data(sample_data)