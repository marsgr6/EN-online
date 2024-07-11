import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import missingno as msno
import ipywidgets as widgets
import ipywidgets
import warnings

def plot_data(all_data):

    @ipywidgets.interact
    def select_data(ds=list(all_data.keys()),
                    plot_type=['bars', 'boxes', 'ridges', 'histogram', 
                          'density 1', 'density 2', 'scatter',
                          'catplot', 'missingno', 'correlation', 'clustermap',
                          'pairplot', 'regression'
                         ],
                    risk_it_all=False,
            ):
        data = all_data[ds]

        for col in data.select_dtypes(include = 'category').columns:
            data[col] = data[col].astype('str')

        if plot_type == "correlation":
            data_c = data.drop(data.columns[data.nunique() == 1],axis=1)
            corr_matrix = data_c.corr()
            sns.clustermap(corr_matrix, annot=True)
            plt.show()

        if plot_type == "clustermap":
            @ipywidgets.interact
            def plot(z_score=[None, 0, 1], standard_scale=[None, 0, 1]):
                data_c = data.drop(
                    data.columns[data.nunique() == 1],axis=1).select_dtypes(
                    include = 'number')
                sns.clustermap(data_c, annot=True, 
                    standard_scale=standard_scale,
                    z_score=z_score
                    )
                plt.show()

        if plot_type == "pairplot":
            @ipywidgets.interact
            def plot(hue_var=data.select_dtypes(include = 'object').columns,
                kind=["kde", "hist"]
                ):
                sns.pairplot(data, hue=hue_var, diag_kind="kde")
                plt.show()

        if plot_type == 'missingno':
            @ipywidgets.interact
            def plot(tplot=["matrix", "bars", "heatmap", "dendrogram"]):
                if tplot=="matrix":
                    msno.matrix(data)
                if tplot=="bars":
                    msno.bar(data)
                if tplot=="heatmap":
                    msno.heatmap(data)
                if tplot=="dendrogram":
                    msno.dendrogram(data)
                plt.show()

        if risk_it_all:
            if plot_type == 'bars':
                @ipywidgets.interact
                def plot(var_x=data.columns,
                         hue=data.columns,
                         tplot=["bars", "heatmap"]
                        ):          # categorical univariate plot
                    sns.reset_defaults()
                    plt.figure(figsize=(8,6))
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
                        _ = plt.ylabel(var_x)
                    plt.show();

            if plot_type == 'boxes':
                    @ipywidgets.interact
                    def plot(var_x=data.columns,
                             var_y=data.columns, 
                             hue=data.columns,
                             tplot=["boxplot", "lineplot", "violin"]
                            ):
                        sns.reset_defaults()
                        palette = sns.color_palette("Set2", 12)  # If you have more than 12 categories change this
                        plt.figure(figsize=(8,6))
                        if tplot == "boxplot":
                            sns.boxplot(data=data, x=var_x, y=var_y, hue=hue, palette=palette);
                        if tplot == "violin":
                            if len(np.unique(data[hue]) == 2):
                                sns.violinplot(data=data, x=var_x, y=var_y, hue=hue, split=True,
                                               common_norm=False,
                                               cut=0, palette=palette, inner="quartil");
                            else: 
                                sns.violinplot(data=data, x=var_x, y=var_y, hue=hue, common_norm=False,
                                               cut=0, palette=palette, inner="quartil")
                        if tplot == "lineplot":
                            sns.lineplot(data=data, x=var_x, y=var_y, hue=hue,
                                         err_style="bars", errorbar=('ci', 68), estimator='mean', palette=palette)
                        plt.show();

            if plot_type == 'ridges':
                    @ipywidgets.interact
                    def plot(var_x=data.columns,
                             var_y=data.columns, # change to float if necessary
                             hue_var=data.columns,
                             no_hue=False,
                            ):
                        if no_hue: hue_var = var_x
                        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
                        palette = sns.color_palette("Set2", 12)  # If you have more than 12 categories change this
                        g = sns.FacetGrid(data, palette=palette, row=var_x, hue=hue_var, aspect=6, height=1.5)
                        g.map_dataframe(sns.kdeplot, x=var_y, cut=0, fill=True, alpha=0.5)
                        #g.map_dataframe(sns.kdeplot, x=var_y, cut=0, color='black', alpha=0.5)
                        g.add_legend()
                        plt.show();

            if plot_type == 'histogram':
                    @ipywidgets.interact
                    def plot(var_x=data.columns, # change to float if necessary
                             hue_var=data.columns,
                             multiple=["layer", "dodge", "stack", "fill"],
                             stat=["count", "probability", "percent", "density"],
                             element=["bars", "step", "poly"],
                             common_norm=False,
                             cumulative=False,
                            ):
                        sns.reset_defaults()
                        plt.figure(figsize=(8,6))
                        palette = sns.color_palette("Set2", 12)  # If you have more than 12 categories change this
                        sns.histplot(data=data, x=var_x, hue=hue_var, stat=stat, element=element, 
                                     cumulative=cumulative, multiple=multiple, 
                                     common_norm=common_norm)
                        plt.show();

            if plot_type == 'density 1':
                    @ipywidgets.interact
                    def plot(var_x=data.columns, # change to float if necessary
                             hue_var=data.columns,
                             multiple=["layer", "stack", "fill"],
                             common_norm=False,
                             cumulative=False,
                            ):
                        sns.reset_defaults()
                        plt.figure(figsize=(8,6))
                        palette = sns.color_palette("Set2", 12)  # If you have more than 12 categories change this
                        sns.kdeplot(data=data, x=var_x, hue=hue_var, 
                                    multiple=multiple, cut=0, fill=True,
                                    common_norm=common_norm, cumulative=cumulative)
                        plt.show();

            if plot_type == 'density 2':
                @ipywidgets.interact
                def plot(var_x=data.columns, 
                         var_y=data.columns, 
                         hue_var=data.columns,
                         col_var=data.columns,
                         kind=["hist", "kde", "ecdf"],
                         common_norm=False,
                         cumulative=False,
                         facet=False,
                         rug=False,
                        ):
                    sns.reset_defaults()
                    plt.figure(figsize=(8,6))
                    palette = sns.color_palette("Set2", 12)  # If you have more than 12 categories change this
                    if var_x != var_y:
                        if facet:
                            sns.displot(data=data, x=var_x, y=var_y, hue=hue_var, 
                                        kind=kind, col=col_var, rug=rug,
                                        common_norm=common_norm, cumulative=cumulative)
                        else:
                            sns.displot(data=data, x=var_x, y=var_y, hue=hue_var, 
                                        kind=kind, rug=rug,
                                        common_norm=common_norm, cumulative=cumulative)
                    else:
                        if facet:
                            sns.displot(data=data, x=var_x, hue=hue_var, 
                                        kind=kind, col=col_var, rug=rug,
                                        common_norm=common_norm, cumulative=cumulative)
                        else:
                            sns.displot(data=data, x=var_x, hue=hue_var, 
                                        kind=kind, rug=rug,
                                        common_norm=common_norm, cumulative=cumulative)

                    plt.show();

            if plot_type == 'scatter':
                    """
                    Colab has an AI copilot you can ask to help you with the code
                    For example: 
                    prompt to colab: add a slider for alpha in the scatter plot and add a style var
                    """
                    @ipywidgets.interact
                    def plot(var_x=data.columns,
                             var_y=data.columns, # change to float if necessary
                             hue=data.columns,
                             style=data.columns,
                             size=data.columns,
                             alpha=widgets.FloatSlider(value=0.5, min=0, max=1, step=0.01),
                             use_style=False,
                            ):
                        sns.reset_defaults()
                        plt.figure(figsize=(8,6))
                        if use_style:
                            sns.scatterplot(data=data, x=var_x, y=var_y, 
                                            hue=hue, style=style, alpha=alpha, size=size);
                        else:
                            sns.scatterplot(data=data, x=var_x, y=var_y, 
                                            hue=hue, alpha=alpha, size=size);
                        plt.show()

            if plot_type == 'catplot':
                """
                Colab has an AI copilot you can ask to help you with the code
                For example: 
                prompt to colab: add a slider for alpha in the scatter plot and add a style var
                """
                @ipywidgets.interact
                def plot(var_x=data.columns,
                         var_y=data.columns, # change to float if necessary
                         hue=data.columns,
                         col=data.columns,
                         kind=["strip", "swarm"], 
                         facet=False,
                        ):
                    sns.reset_defaults()
                    plt.figure(figsize=(8,6))
                    if facet:
                        sns.catplot(data=data, x=var_x, y=var_y, col=col, hue=hue, kind=kind);
                    else:
                        sns.catplot(data=data, x=var_x, y=var_y, hue=hue, kind=kind);
                    plt.show()

            if plot_type == 'regression':
                """
                Colab has an AI copilot you can ask to help you with the code
                For example: 
                prompt to colab: add a slider for alpha in the scatter plot and add a style var
                """
                @ipywidgets.interact
                def plot(var_x=data.columns,
                         var_y=data.columns, # change to float if necessary
                         hue_var=data.columns,
                         order=[1, 2, 3], 
                         ci=[68, 95, 99, 0],
                         use_hue=True,
                        ):
                    sns.reset_defaults()
                    plt.figure(figsize=(8,6))
                    if use_hue:
                        sns.lmplot(data=data, x=var_x, y=var_y, hue=hue, order=order, ci=ci)
                    else:
                        sns.lmplot(data=data, x=var_x, y=var_y, order=order, ci=ci)
                    plt.show()

        else:
            if plot_type == 'bars':
                @ipywidgets.interact
                def plot(var_x=data.select_dtypes(include = 'object').columns,
                         hue=data.select_dtypes(include = 'object').columns,
                         tplot=["bars", "heatmap"]
                        ):          # categorical univariate plot
                    sns.reset_defaults()
                    plt.figure(figsize=(8,6))
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
                        _ = plt.ylabel(var_x)
                    plt.show();

            if plot_type == 'boxes':
                    @ipywidgets.interact
                    def plot(var_x=data.select_dtypes(include = 'object').columns,
                             var_y=data.select_dtypes(include = 'number').columns, 
                             hue=data.select_dtypes(include = 'object').columns,
                             tplot=["boxplot", "lineplot", "violin"],
                             no_hue=False,
                            ):
                        sns.reset_defaults()
                        palette = sns.color_palette("Set2", 12)  # If you have more than 12 categories change this
                        plt.figure(figsize=(8,6))
                        if tplot == "boxplot":
                            if not no_hue:
                                sns.boxplot(data=data, x=var_x, y=var_y, hue=hue, palette=palette);
                            else:
                                sns.boxplot(data=data, x=var_x, y=var_y, palette=palette);
                        if tplot == "violin":
                            if len(np.unique(data[hue]) == 2):
                                if no_hue:
                                    sns.violinplot(data=data, x=var_x, y=var_y, common_norm=False,
                                               cut=0, palette=palette, inner="quartil")
                                else:
                                    sns.violinplot(data=data, x=var_x, y=var_y, hue=hue, split=True,
                                               common_norm=False,
                                               cut=0, palette=palette, inner="quartil");
                            else: 
                                if no_hue:
                                    sns.violinplot(data=data, x=var_x, y=var_y, common_norm=False,
                                               cut=0, palette=palette, inner="quartil")
                                else:
                                    sns.violinplot(data=data, x=var_x, y=var_y, hue=hue, common_norm=False,
                                               cut=0, palette=palette, inner="quartil")

                        if tplot == "lineplot":
                            if not no_hue:
                                sns.lineplot(data=data, x=var_x, y=var_y,
                                         err_style="bars", errorbar=('ci', 68), estimator='mean', palette=palette)
                            else:
                                sns.lineplot(data=data, x=var_x, y=var_y, hue=hue,
                                            err_style="bars", errorbar=('ci', 68), estimator='mean', palette=palette)
                        plt.show();

            if plot_type == 'ridges':
                    @ipywidgets.interact
                    def plot(var_x=data.select_dtypes(include = 'object').columns,
                             var_y=data.select_dtypes(include = 'number').columns, # change to float if necessary
                             hue_var=data.select_dtypes(include = 'object').columns,
                             no_hue=False,
                            ):
                        if no_hue: hue_var = var_x
                        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
                        palette = sns.color_palette("Set2", 12)  # If you have more than 12 categories change this
                        g = sns.FacetGrid(data, palette=palette, row=var_x, hue=hue_var, aspect=6, height=1.5)
                        g.map_dataframe(sns.kdeplot, x=var_y, cut=0, fill=True, alpha=0.5)
                        #g.map_dataframe(sns.kdeplot, x=var_y, cut=0, color='black', alpha=0.5)
                        g.add_legend()
                        plt.show();

            if plot_type == 'histogram':
                    @ipywidgets.interact
                    def plot(var_x=data.select_dtypes(include = 'number').columns, # change to float if necessary
                             hue_var=data.select_dtypes(include = 'object').columns,
                             multiple=["layer", "dodge", "stack", "fill"],
                             stat=["count", "probability", "percent", "density"],
                             element=["bars", "step", "poly"],
                             common_norm=False,
                             cumulative=False,
                            ):
                        sns.reset_defaults()
                        plt.figure(figsize=(8,6))
                        palette = sns.color_palette("Set2", 12)  # If you have more than 12 categories change this
                        sns.histplot(data=data, x=var_x, hue=hue_var, stat=stat, element=element, 
                                     cumulative=cumulative, multiple=multiple, 
                                     common_norm=common_norm)
                        plt.show();

            if plot_type == 'density 1':
                    @ipywidgets.interact
                    def plot(var_x=data.select_dtypes(include = 'number').columns, # change to float if necessary
                             hue_var=data.select_dtypes(include = 'object').columns,
                             multiple=["layer", "stack", "fill"],
                             common_norm=False,
                             cumulative=False,
                            ):
                        sns.reset_defaults()
                        plt.figure(figsize=(8,6))
                        palette = sns.color_palette("Set2", 12)  # If you have more than 12 categories change this
                        sns.kdeplot(data=data, x=var_x, hue=hue_var, 
                                    multiple=multiple, cut=0, fill=True,
                                    common_norm=common_norm, cumulative=cumulative)
                        plt.show();

            if plot_type == 'density 2':
                @ipywidgets.interact
                def plot(var_x=data.select_dtypes(include = 'number').columns, 
                         var_y=data.select_dtypes(include = 'number').columns, 
                         hue_var=data.select_dtypes(include = 'object').columns,
                         col_var=data.select_dtypes(include = 'object').columns,
                         kind=["hist", "kde", "ecdf"],
                         common_norm=False,
                         cumulative=False,
                         facet=False,
                         rug=False,
                        ):
                    sns.reset_defaults()
                    plt.figure(figsize=(8,6))
                    palette = sns.color_palette("Set2", 12)  # If you have more than 12 categories change this
                    if var_x != var_y:
                        if facet:
                            sns.displot(data=data, x=var_x, y=var_y, hue=hue_var, 
                                        kind=kind, col=col_var, rug=rug,
                                        common_norm=common_norm, cumulative=cumulative)
                        else:
                            sns.displot(data=data, x=var_x, y=var_y, hue=hue_var, 
                                        kind=kind, rug=rug,
                                        common_norm=common_norm, cumulative=cumulative)
                    else:
                        if facet:
                            sns.displot(data=data, x=var_x, hue=hue_var, 
                                        kind=kind, col=col_var, rug=rug,
                                        common_norm=common_norm, cumulative=cumulative)
                        else:
                            sns.displot(data=data, x=var_x, hue=hue_var, 
                                        kind=kind, rug=rug,
                                        common_norm=common_norm, cumulative=cumulative)

                    plt.show();

            if plot_type == 'scatter':
                    """
                    Colab has an AI copilot you can ask to help you with the code
                    For example: 
                    prompt to colab: add a slider for alpha in the scatter plot and add a style var
                    """
                    @ipywidgets.interact
                    def plot(var_x=data.select_dtypes(include = 'number').columns,
                             var_y=data.select_dtypes(include = 'number').columns, # change to float if necessary
                             hue=data.select_dtypes(include = 'object').columns,
                             style=data.select_dtypes(include = 'object').columns,
                             size=data.select_dtypes(include = 'number').columns,
                             alpha=widgets.FloatSlider(value=0.5, min=0, max=1, step=0.01),
                             use_style=False,
                            ):
                        sns.reset_defaults()
                        plt.figure(figsize=(8,6))
                        if use_style:
                            sns.scatterplot(data=data, x=var_x, y=var_y, 
                                            hue=hue, style=style, alpha=alpha, size=size);
                        else:
                            sns.scatterplot(data=data, x=var_x, y=var_y, 
                                            hue=hue, alpha=alpha, size=size);
                        plt.show()

            if plot_type == 'catplot':
                """
                Colab has an AI copilot you can ask to help you with the code
                For example: 
                prompt to colab: add a slider for alpha in the scatter plot and add a style var
                """
                @ipywidgets.interact
                def plot(var_x=data.select_dtypes(include = 'number').columns,
                         var_y=data.select_dtypes(include = 'object').columns, # change to float if necessary
                         hue=data.select_dtypes(include = 'object').columns,
                         col=data.select_dtypes(include = 'object').columns,
                         kind=["strip", "swarm"], 
                         facet=False,
                        ):
                    sns.reset_defaults()
                    plt.figure(figsize=(8,6))
                    if facet:
                        sns.catplot(data=data, x=var_x, y=var_y, col=col, hue=hue, kind=kind);
                    else:
                        sns.catplot(data=data, x=var_x, y=var_y, hue=hue, kind=kind);
                    plt.show()


            if plot_type == 'regression':
                """
                Colab has an AI copilot you can ask to help you with the code
                For example: 
                prompt to colab: add a slider for alpha in the scatter plot and add a style var
                """
                @ipywidgets.interact
                def plot(var_x=data.select_dtypes(include = 'number').columns,
                         var_y=data.select_dtypes(include = 'number').columns, # change to float if necessary
                         hue=data.select_dtypes(include = 'object').columns,
                         order=[1, 2, 3], 
                         ci=[68, 95, 99, 0],
                         use_hue=True,
                        ):
                    sns.reset_defaults()
                    plt.figure(figsize=(8,6))
                    if use_hue:
                        sns.lmplot(data=data, x=var_x, y=var_y, hue=hue, order=order, ci=ci)
                    else:
                        sns.lmplot(data=data, x=var_x, y=var_y, order=order, ci=ci)
                    plt.show()