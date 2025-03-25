import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ipywidgets as widgets
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def plot_data(all_data):
    """Interactive plotting function for exploring datasets."""
    
    # Define plot types
    PLOT_TYPES = [
        'bars', 'boxes', 'ridges', 'histogram', 'density 1', 'density 2', 
        'scatter', 'catplot', 'missingno', 'correlation', 'clustermap', 
        'pairplot', 'regression'
    ]

    @widgets.interact(ds=list(all_data.keys()), plot_type=PLOT_TYPES, risk_it_all=False)
    def select_data(ds, plot_type, risk_it_all):
        data = all_data[ds].copy()
        
        # Convert categorical columns to strings
        for col in data.select_dtypes(include='category').columns:
            data[col] = data[col].astype('str')

        # Helper function to set default plot settings
        def set_plot_defaults(figsize=(8, 6)):
            sns.reset_defaults()
            plt.figure(figsize=figsize)

        # Define palette
        PALETTE = sns.color_palette("Set2", 12)

        # Non-interactive plot types
        if plot_type == "correlation":
            data_c = data.drop(data.columns[data.nunique() == 1], axis=1)
            corr_matrix = data_c.select_dtypes(include=np.number).corr()
            sns.clustermap(corr_matrix, annot=True)
            plt.show()
            return

        elif plot_type == "clustermap":
            @widgets.interact(z_score=[None, 0, 1], standard_scale=[None, 0, 1])
            def plot_clustermap(z_score, standard_scale):
                data_c = data.drop(data.columns[data.nunique() == 1], axis=1)
                numeric_data = data_c.select_dtypes(include='number').dropna()
                sns.clustermap(numeric_data, annot=True, standard_scale=standard_scale, z_score=z_score)
                plt.show()

        elif plot_type == "pairplot":
            @widgets.interact(hue_var=data.select_dtypes(include='object').columns, kind=["kde", "hist"])
            def plot_pairplot(hue_var, kind):
                sns.pairplot(data, hue=hue_var, diag_kind=kind)
                plt.show()

        elif plot_type == "missingno":
            @widgets.interact(tplot=["matrix", "bars", "heatmap", "dendrogram"])
            def plot_missingno(tplot):
                plot_functions = {
                    "matrix": msno.matrix,
                    "bars": msno.bar,
                    "heatmap": msno.heatmap,
                    "dendrogram": msno.dendrogram
                }
                plot_functions[tplot](data)
                plt.show()

        # Interactive plot types with risk_it_all toggle
        else:
            # Bars: x must be categorical, hue can be categorical
            if plot_type == 'bars':
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                @widgets.interact(var_x=x_cols, hue=hue_cols, tplot=["bars", "heatmap"])
                def plot_bars(var_x, hue, tplot):
                    set_plot_defaults()
                    if tplot == "bars":
                        ax = sns.countplot(data=data, x=var_x, hue=hue)
                        for container in ax.containers:
                            ax.bar_label(container)
                    else:
                        df_2dhist = pd.DataFrame({
                            x_label: grp[var_x].value_counts()
                            for x_label, grp in data.groupby(hue)
                        })
                        sns.heatmap(df_2dhist, cmap='viridis', annot=True, cbar=False, fmt='.0f')
                        plt.xlabel(hue)
                        plt.ylabel(var_x)
                    plt.show()

            # Boxes: x categorical, y numeric, hue categorical
            elif plot_type == 'boxes':
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                y_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                @widgets.interact(var_x=x_cols, var_y=y_cols, hue=hue_cols, tplot=["boxplot", "lineplot", "violin"], no_hue=False)
                def plot_boxes(var_x, var_y, hue, tplot, no_hue):
                    set_plot_defaults()
                    hue_param = None if (no_hue and not risk_it_all) else hue
                    if tplot == "boxplot":
                        sns.boxplot(data=data, x=var_x, y=var_y, hue=hue_param, palette=PALETTE)
                    elif tplot == "violin":
                        if len(np.unique(data[hue])) == 2 and not no_hue:
                            sns.violinplot(data=data, x=var_x, y=var_y, hue=hue, split=True,
                                           common_norm=False, cut=0, palette=PALETTE, inner="quartile")
                        else:
                            sns.violinplot(data=data, x=var_x, y=var_y, hue=hue_param,
                                           common_norm=False, cut=0, palette=PALETTE, inner="quartile")
                    elif tplot == "lineplot":
                        sns.lineplot(data=data, x=var_x, y=var_y, hue=hue_param,
                                     err_style="bars", errorbar=('ci', 68), estimator='mean', palette=PALETTE)
                    plt.show()

            # Ridges: x categorical, y numeric, hue categorical
            elif plot_type == 'ridges':
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                y_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                @widgets.interact(var_x=x_cols, var_y=y_cols, hue_var=hue_cols, no_hue=False)
                def plot_ridges(var_x, var_y, hue_var, no_hue):
                    hue_param = var_x if no_hue else hue_var
                    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth': 2})
                    g = sns.FacetGrid(data, palette=PALETTE, row=var_x, hue=hue_param, aspect=6, height=1.5)
                    g.map_dataframe(sns.kdeplot, x=var_y, cut=0, fill=True, alpha=0.5)
                    g.add_legend()
                    plt.show()

            # Histogram: x numeric, hue categorical
            elif plot_type == 'histogram':
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                @widgets.interact(var_x=x_cols, hue_var=hue_cols, multiple=["layer", "dodge", "stack", "fill"],
                                  stat=["count", "probability", "percent", "density"], element=["bars", "step", "poly"],
                                  common_norm=False, cumulative=False)
                def plot_histogram(var_x, hue_var, multiple, stat, element, common_norm, cumulative):
                    set_plot_defaults()
                    sns.histplot(data=data, x=var_x, hue=hue_var, stat=stat, element=element,
                                 cumulative=cumulative, multiple=multiple, common_norm=common_norm, palette=PALETTE)
                    plt.show()

            # Density 1: x numeric, hue categorical
            elif plot_type == 'density 1':
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                @widgets.interact(var_x=x_cols, hue_var=hue_cols, multiple=["layer", "stack", "fill"],
                                  common_norm=False, cumulative=False)
                def plot_density1(var_x, hue_var, multiple, common_norm, cumulative):
                    set_plot_defaults()
                    sns.kdeplot(data=data, x=var_x, hue=hue_var, multiple=multiple, cut=0, fill=True,
                                common_norm=common_norm, cumulative=cumulative, palette=PALETTE)
                    plt.show()

            # Density 2: x numeric, y numeric (optional), hue categorical, col categorical
            elif plot_type == 'density 2':
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                y_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                col_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                @widgets.interact(var_x=x_cols, var_y=y_cols, hue_var=hue_cols, col_var=col_cols,
                                  kind=["hist", "kde", "ecdf"], common_norm=False, cumulative=False, facet=False, rug=False)
                def plot_density2(var_x, var_y, hue_var, col_var, kind, common_norm, cumulative, facet, rug):
                    set_plot_defaults()
                    kwargs = {'hue': hue_var, 'kind': kind, 'rug': rug, 'common_norm': common_norm, 'cumulative': cumulative}
                    if facet:
                        sns.displot(data=data, x=var_x, y=var_y if var_x != var_y else None, col=col_var, **kwargs)
                    else:
                        sns.displot(data=data, x=var_x, y=var_y if var_x != var_y else None, **kwargs)
                    plt.show()

            # Scatter: x numeric, y numeric, hue categorical, style categorical, size numeric
            elif plot_type == 'scatter':
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                y_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                style_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                size_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                @widgets.interact(var_x=x_cols, var_y=y_cols, hue=hue_cols, style=style_cols, size=size_cols,
                                  alpha=widgets.FloatSlider(value=0.5, min=0, max=1, step=0.01), use_style=False)
                def plot_scatter(var_x, var_y, hue, style, size, alpha, use_style):
                    set_plot_defaults()
                    kwargs = {'x': var_x, 'y': var_y, 'hue': hue, 'alpha': alpha, 'size': size}
                    if use_style:
                        kwargs['style'] = style
                    sns.scatterplot(data=data, **kwargs)
                    plt.show()

            # Catplot: x can be numeric or categorical, y categorical, hue categorical, col categorical
            elif plot_type == 'catplot':
                x_cols = data.columns if risk_it_all else data.columns  # Allow both numeric and categorical
                y_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                col_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                @widgets.interact(var_x=x_cols, var_y=y_cols, hue=hue_cols, col=col_cols, kind=["strip", "swarm"], facet=False)
                def plot_catplot(var_x, var_y, hue, col, kind, facet):
                    set_plot_defaults()
                    if facet:
                        sns.catplot(data=data, x=var_x, y=var_y, col=col, hue=hue, kind=kind)
                    else:
                        sns.catplot(data=data, x=var_x, y=var_y, hue=hue, kind=kind)
                    plt.show()

            # Regression: x numeric, y numeric, hue categorical
            elif plot_type == 'regression':
                x_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                y_cols = data.columns if risk_it_all else data.select_dtypes(include='number').columns
                hue_cols = data.columns if risk_it_all else data.select_dtypes(include='object').columns
                @widgets.interact(var_x=x_cols, var_y=y_cols, hue=hue_cols, order=[1, 2, 3], ci=[68, 95, 99, 0], use_hue=True)
                def plot_regression(var_x, var_y, hue, order, ci, use_hue):
                    set_plot_defaults()
                    sns.lmplot(data=data, x=var_x, y=var_y, hue=hue if use_hue else None, order=order, ci=ci)
                    plt.show()