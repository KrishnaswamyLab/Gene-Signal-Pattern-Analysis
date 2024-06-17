# Gene Signal Pattern Analysis
### Mapping the gene space at single-cell resolution

Gene Signal Pattern Analysis is a Python package for mapping the gene space from single-cell data. For a detailed explanation of GSPA and potential downstream application, see:

[**Mapping the gene space at single-cell resolution with Gene Signal Pattern Analysis**. Aarthi Venkat, Sam Leone, Scott E. Youlten, Eric Fagerberg, John Attanasio, Nikhil S. Joshi, Michael Perlmutter, Smita Krishnaswamy.](https://www.biorxiv.org/content/10.1101/2023.11.26.568492v1)

By considering gene expression values as signals on the cell-cell graph, GSPA enables complex analyses of gene-gene relationships, including gene cluster analysis, cell-cell communication, and patient manifold learning from gene-gene graphs.

### Installation

```
pip install gspa
```

### Requirements

GSPA requires Python >= 3.6. All other requirements are automatically installed by ``pip`` (see also requirements.txt).

The following have been tested: Python 3.6.18 (graphtools 1.5.3, tensorflow 2.6.2, keras 2.6.0, numpy 1.19.5, sklearn 0.24.2, scipy 1.5.4, tqdm 4.64.1, scanpy 1.7.2, phate 1.0.11) and Python 3.8.18 (graphtools 1.5.3, tensorflow 2.13.0, keras 2.13.1, numpy 1.22.4, sklearn 1.3.2, scipy 1.10.1, tqdm 4.66.4, scanpy 1.9.3, phate 1.0.11)

### Usage example

```
    import numpy as np
    import gspa
    
    # Create toy data
    n_cells = 1000
    n_genes = 50
    data = np.random.normal(size=(n_cells, n_genes))

    # GSPA operator constructs wavelet dictionary
    gspa_op = gspa.GSPA()
    gspa_op.construct_graph(data)
    gspa_op.build_diffusion_operator()
    gspa_op.build_wavelet_dictionary()

    # Embed gene signals from wavelet dictionary
    gene_signals = data.T # embed all measured genes
    gene_ae, gene_pc = gspa_op.get_gene_embeddings(gene_signals)
    gene_localization = gspa_op.calculate_localization()
```

See `GSPA_example.ipynb` at [GitHub](https://github.com/KrishnaswamyLab/Gene-Signal-Pattern-Analysis) for test run on simulated single-cell data.
