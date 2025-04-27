

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl

    return mo, np, plt


@app.cell
def softmax_description(mo):
    mo.md(
        r"""
        # LLM adjustment parameters
        ## SoftMax

        We define $softmax$ for a vector $x \in \Re^k$ as:

        $$
        softmax(x_i)= p_i= \frac{e^x}{\sum_{j=1}^{k}e^{x_j}}
        $$
        """
    )
    return


@app.cell
def temperature_definition(mo, temperature_slider):
    t=temperature_slider.value
    mo.md(
        rf"""
        ## Temperature
        We introduce tempaerature with the following formula:

        $$
        p_i(T)= \frac{{e^{{x_i \over T}}}}{{\sum_{{j=1}}^{{k}}e^{{x_j \over T}}}}
        $$
    """+rf"""
        So for T={t:0.3}

        $$
        p_i({t:0.3})= \frac{{e^{{x_i\over {t:0.3}}}}}{{\sum_{{j=1}}^{{k}}e^{{x_j \over {t:0.3}}}}}
        $$
    
        The logits are:
        """

    
    )
    return


@app.cell
def temperature_logits(np):
    _min = -4
    _max = 4
    _temperature_num_options = 20
    rng = np.random.default_rng(seed=12345)
    logits = (_max - _min) * rng.random(_temperature_num_options) + _min
    logits
    return logits, rng


@app.cell
def temperature_slider(mo, np):
    temperature_slider = mo.ui.slider(steps=np.logspace(-1, 1, 41), label="Temperature", full_width=True, value=1)
    return (temperature_slider,)


@app.cell
def _(np):
    def soft_max(x, temp=1):
        ps = np.exp(x/temp)
        ps /= np.sum(ps)
        return ps

    return (soft_max,)


@app.cell
def temperature_ui(mo, temperature_slider):
    mo.hstack([temperature_slider, mo.md(f"Has value: {temperature_slider.value:0.2f}")])
    return


@app.cell
def temperature_graph(logits, np, plt, soft_max, temperature_slider):
    def draw_distribution(logits, t=1):
        # Indices for the x-axis (e.g., 0, 1, 2, 3, 4)
        distribution = soft_max(logits,t)
        x = np.arange(len(distribution))
        # Create a figure and axes
        fig, axs= plt.subplots(1,2, figsize=(15, 5))
    
        for idx , ax in enumerate(axs):
            # Plot using stem
            markerline, stemlines, baseline = ax.stem(x, distribution)
            plt.setp(markerline, markersize=8)
            plt.setp(stemlines, linewidth=2)
            baseline.set_visible(False)
        
        
            # Only horizontal grid lines
            ax.grid(True, which="both", axis='y')
        
            # Remove x-axis ticks and label
            ax.set_xticks([])    # no ticks
            ax.set_xlabel('')    # no label
        
        # Y-axis label and title
        ax.set_title('Distribution ')
        axs[0].set_ylabel('Probability (log scale)')
        axs[1].set_ylabel('Probability')
        # Set log scale for y-axis
        axs[0].set_yscale('log')
        return fig,axs

    _fig, _axs = draw_distribution(logits, temperature_slider.value)

    _fig
    return


@app.cell
def _(mo, rng):
    top_p = mo.ui.slider(start=0, stop=1, step=0.01, label="Top P", full_width=True, value=1)
    _num_samples = 20
    top_p_dist = rng.random(_num_samples)
    return top_p, top_p_dist


@app.cell(hide_code=True)
def top_p_graph(mo, top_p):
    _description = mo.md(
        """
        ## Top P (nucleus) sampling

        Sum the top most likely outcomes until you get to P.

        If you take top 0 ... you sample just the most probable

        If you take top 1 ... you sample from all probabilites
        """
    )

    _slider = mo.hstack([top_p, mo.md(f"Has value: {top_p.value:0.2f}")])

    mo.vstack([_description, _slider])
    return


@app.cell
def _(np, plt, top_p, top_p_dist):


    _dist = top_p_dist / top_p_dist.sum()
    def to_top_p( dist, p=1):
        pairs = {x for x in enumerate(dist)}
        sort_by_prob = sorted(pairs, key = lambda pair: -pair[1])
        new_probs = np.zeros(len(dist), dtype=np.float64)
        sum = sort_by_prob[0][1]
        new_probs[sort_by_prob[0][0]]=sum
        for pos, prob in sort_by_prob[1:]:
            if sum + prob < p:
                new_probs[pos] = prob
                sum += prob
            else:
                break
        return new_probs / new_probs.sum()

    def draw_top_p(distribution, p=1):
        fig, axs= plt.subplots(1,2, figsize=(15, 5))
        x = np.arange(len(distribution))    
        for idx in [0,1]:
            ax = axs[idx]
            if idx == 0:
                dist = distribution
            else:
                dist = to_top_p(distribution,p)
            # Plot using stem
            markerline, stemlines, baseline = ax.stem(x, dist)
            plt.setp(markerline, markersize=8)
            plt.setp(stemlines, linewidth=2)
            baseline.set_visible(False)
        
        
            # Only horizontal grid lines
            ax.grid(True, which="both", axis='y')
        
            # Remove x-axis ticks and label
            ax.set_xticks([])    # no ticks
            ax.set_xlabel('')    # no label
            ax.set_ylabel("Probability")
    
        # Y-axis label and title
        axs[0].set_title('Initial Distribution')
        axs[1].set_title(f'Distribution Top P={top_p.value}')
        # Set log scale for y-axis

        return axs[0]
    draw_top_p( _dist, top_p.value)
    return


@app.cell
def pentalty_description(mo):
    mo.md(
        r"""
        ## Frequency and Presence Penalty

        Logit adjustment:

        $$
        z_i^\prime = z_i - (ppenalty \times p_i) -(fpentalty \times f_i)
        $$

        where:

        - $p_i$ is $0$ if the token has not been already seen and $1$ otherwise.
        - $f_i$ is the number of times token $i$ has already been seen in the output.
        - $ppenalty$ is the presence penalty $[-2,2]$.
        - $fpenalty$ is the frequency penalty $[-2,2]$.
        """
    )
    return


if __name__ == "__main__":
    app.run()
