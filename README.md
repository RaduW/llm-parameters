# LLM Parameters

An illustration of the parameters used in LLMs to control 'creativity'.


This is inspired by [ritvikmath's](https://www.youtube.com/@ritvikmath)  excellent video on the subject
[The 4 Must-Know LLM Parameters and the Intuitive Math Behind Them](https://www.youtube.com/watch?v=33kb37NYOTc).



[Here's the link to the notebook](https://raduw.github.io/llm-parameters/)


# Site generation

To generate the site, run the following command:

```zsh
marimo export html-wasm notebook.py -o docs/index.html
```

To run the notebook locally using python, use the following command:

```zsh
marimo run notebook.py
```

To run the notebook locally using wasm, generate the site.
cd into the directory where the site was generated, and run:

```zsh
python -m http.server 8000
```

Then open your browser and navigate to `http://localhost:8000`.
