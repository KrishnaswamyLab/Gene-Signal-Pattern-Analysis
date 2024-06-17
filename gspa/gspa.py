import tasklogger
import graphtools
from . import graphs, wavelets, embedding
import numpy as np
from tqdm import tqdm
from scipy import sparse, spatial
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
_logger = tasklogger.get_tasklogger("graphtools")

class GSPA:
    """GSPA operator which performs gene dimensionality reduction.

    Gene Signal Pattern Analysis (GSPA) considers genes as signals on a cell-cell graph, enabling mapping the gene space for complex gene-level analyses, including gene cluster analysis, cell-cell communication, and patient manifold learning from gene-gene graphs (Venkat et al. [1]_).

    Parameters
    ----------
    graph : graphtools.Graph, optional, default: None
        Cell-cell affinity graph. If None, `construct_graph` function will need to be run to construct graph from data directly.
    diffusion_operator : array-like, shape=[n_samples, n_samples], default: None
        Cell-cell diffusion operator. If None `build_diffusion_operator` will need to be run based on the cell-cell affinity graph.
    qr_decompose : boolean, default: True
        If True, composes reduced wavelet dictionary with QR decomposition
    qr_epsilon: float, optional, default: 1e-3
        If qr_decompose is True, qr_epsilon determines threshold for QR decomposition
    wavelet_J: int, optional, default: -1
        Maximum number of scales J. If -1, uses J=log(number of cells) based on Tong et al. [2]_.
    wavelet_power: int, optional, default: 2
        Geometric seqence of ratio wavelet_ower for wavelet transforms.
    embedding_dim: int, optional, default: 128
        Number of dimensions in which genes will be embedded with autoencoder.
    pc_dim: int, optional, default: 2048
        Number of dimensions in which genes will be embedded with PCA.
    random_state: int, default: 42
        Integer random seed for GSPA pipeline.
    verbose: boolean, optional, default: True
        If True, print status messages
    n_jobs: integer, optional, default: -1
        The number of jobs to use for computation. If -1 all CPUS are used.
    perform_condensation: boolean, optional, default: True
        If True, perform graph condensation for large graphs.
    condensation_threshold: int, optional, default: 10000
        If perform_condensation is True, graph condensation occurrs for graphs with more than condensation_threshold cells.
    bc_sample_idx: array-like, shape=[n_samples, 1], default: None
        Batch labels. If provided, bc_sample_idx is used to construct mutual nearest neighbors (MNN) graph for batch correction.
    bc_theta: float, optional, default: 0.95
        If batch labels bc_sample_idx provided, bc_theta is used to parametrize MNN symmetrization.
    activation: string, optional, default: relu
        Activation function in `keras.activations` between layers for autoencoder.
    bias: boolean, optional, default: 1
        If 1, autoencoder layers use bias vector.
    num_layers: int, optional, default: 2
        Number of dense layers within encoder (decoder) of autoencoder.
    dropout: float, optional, default: 0.0
        If dropout > 0, adds dropout layers between dense layers with `dropout` fraction of input units dropped.
    lr: float, optional, default: 0.001
        Learning rate for model Adam optimizer.
    weight_decay: float, optional, default: 0.0
        If set, weight_decay is applied to Adam optimizer.
    epochs: int, optional, default: 100
        Number of epochs for model training.
    val_prop: float, optional, default: 0.05
        Proportion of data heldout for validation set used for early stopping.
    patience: int, optional, default: 10
        Number of epochs with no improvement to validation loss after which training will be stopped.

    Attributes
    ----------
    condensation_groupings: array-like, shape=[n_samples, 1]
        If perform_condensation is True and graph size is greater than condensation_threshold, MS PHATE (Kuchroo et al. [3]_) computes a new node assignment for each cell, where cells are grouped into nodes based on diffusion condensation.
    wavelet_dictionary: array-like, shape=[n_samples, n_wavelet_dictionary_dimensions]
        Stores wavelet dictionary vector for each cell after `build_wavelet_dictionary` is run.
    signals_projected: array-like, shape=[n_features, n_wavelet_dictionary_dimensions]
        Stores gene signals projected onto wavelet dictionary from `get_gene_embeddings`.

    Examples
    ----------

    >>> import gspa
    >>> import scprep
    >>> import numpy as np
    >>> data = np.random.normal(size=(1000, 50)) # fake dataset with 1000 cells and 50 genes
    >>> gspa_op = gspa.GSPA()
    >>> gspa_op.construct_graph(data)
    >>> gspa_op.build_diffusion_operator()
    >>> gspa_op.build_wavelet_dictionary()
    >>> gene_ae, gene_pc = gspa_op.get_gene_embeddings(data.T)
    >>> gene_localization = gspa_op.calculate_localization()
    >>> gene_phate = phate.PHATE().fit_transform(gene_ae)
    >>> scprep.plot.scatter2d(gene_phate, c=gene_localization, cmap='PuBuGn')

    References
    ----------
    .. [1] Venkat A, Leone S, Youlten SE, Fagerberg E, Attansio J, Joshi NS, Perlmutter M, Krishnaswamy S, *Mapping the gene space at single-cell resolution with gene signal pattern analysis* `BioRxiv <https://www.biorxiv.org/content/10.1101/2023.11.26.568492v1>`_.
    .. [2] Tong A, Huguet G, Shung D, Natik A, Kuchroo M, Lajoie G, Wolf G, Krishnaswamy S, *Embedding Signals on Knowledge Graphs with Unbalanced Diffusion Earth Mover's Distance* `arXiv <https://arxiv.org/abs/2107.12334>`_.
    .. [3] Kuchroo et al, *Multiscale PHATE identifies multimodal signatures of COVID-19* `<https://www.nature.com/articles/s41587-021-01186-x>`_.
    """
    
    def __init__(self,
                 graph=None,
                 diffusion_operator=None,
                 qr_decompose=True,
                 qr_epsilon=1e-3,
                 wavelet_J=-1,
                 wavelet_power=2,
                 embedding_dim=128,
                 pc_dim=2048,
                 random_state=42,
                 verbose=True,
                 n_jobs=-1,
                 perform_condensation=True,
                 condensation_threshold=10000,
                 bc_sample_idx=None,
                 bc_theta=0.95,
                 activation='relu',
                 bias=1,
                 num_layers=2,
                 dropout=0.0,
                 lr=0.001,
                 weight_decay=0.0,
                 epochs=100,
                 val_prop=0.05,
                 patience=10,
                 ):

        self.graph = graph
        self.diff_op = diffusion_operator
        self.qr_decompose = qr_decompose
        self.qr_epsilon = qr_epsilon
        self.wavelet_J = wavelet_J
        self.wavelet_power = wavelet_power
        self.embedding_dim = embedding_dim
        self.pc_dim = pc_dim
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.perform_condensation = perform_condensation
        self.condensation_threshold = condensation_threshold
        self.bc_sample_idx = bc_sample_idx
        self.bc_theta = bc_theta
        self.activation = activation
        self.bias = bias
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.val_prop = val_prop
        self.weight_decay = weight_decay
        self.patience = patience

        self.condensation_groupings = None
        self.wavelet_dictionary = None
        self.signals_projected = None

        _logger.set_level(self.verbose)

    def construct_graph(self, data):
        """Constructs cell-cell affinity graph.

        Parameters
        ----------
        data: array-like, shape=[n_samples, n_features]
            input data with `n_samples` samples and `n_features` features. Accepted data types: `numpy.ndarray`, `pd.DataFrame`.
        
        """
        if (data.shape[0] > self.condensation_threshold) & (self.perform_condensation):
            _logger.log_info("Dataset is larger than %s cells. Running graph condensation. Set perform_condensation=False to run exact GSPA." % self.condensation_threshold)
            self.condensation_groupings = graphs.graph_condensation(data, random_state=self.random_state,
                                    n_jobs=self.n_jobs,
                                    condensation_threshold=self.condensation_threshold,
                                    n_pca=self.pc_dim)

            data = graphs.aggregate_signals_over_condensed_nodes(data, self.condensation_groupings)
        if self.bc_sample_idx is None:
            self.graph = graphtools.Graph(data, n_pca=100, random_state=self.random_state, verbose=self.verbose, use_pygsp=True)
        else:
            _logger.log_info(f"bc_sample_idx used for batch correction")
            self.graph = graphtools.Graph(data, n_pca=100, sample_idx=self.bc_sample_idx,
                                 theta=self.bc_theta, random_state=self.random_state, verbose=self.verbose, use_pygsp=True)
        
    def build_diffusion_operator(self):
        """Constructs diffusion operator from graph.
        """
        if self.graph is None:
            raise ValueError('Graph not constructed. Run gspa_op.construct_graph(data) or initialize GSPA operator with graph')
        else:
            self.graph = self.graph.to_pygsp()
            
        Dmin1 = np.diag([1/np.sum(row) for row in self.graph.A])
        self.diff_op = 1/2 * (np.eye(self.graph.N)+self.graph.A@Dmin1)

    def build_wavelet_dictionary(self):
        """Constructs wavelet dictionary from diffusion operator.
        """
        if self.diff_op is None:
            raise ValueError('Diffusion operator not constructed. Run gspa_op.build_diffusion_operator() or initialize GSPA operator with diffusion_operator')
        wavelet_sizes = []

        if self.graph is not None:
            self.graph = self.graph.to_pygsp()

        if sparse.issparse(self.diff_op):
            self.diff_op = self.diff_op.toarray()

        N = self.diff_op.shape[0]
        if self.wavelet_J == -1:
            self.wavelet_J = int(np.log(N))
        I = np.eye(N)
        I = wavelets.normalize(I)
        wavelet_dictionary = [I]
        wavelet_sizes.append(I.shape[1])
        P_j = np.linalg.matrix_power(self.diff_op, self.wavelet_power)
    
        if self.qr_decompose:
            Psi_j_tilde = wavelets.column_subset(I-P_j, epsilon=self.qr_epsilon)
            
            if Psi_j_tilde.shape[1] == 0: 
                _logger.log_info(f"Wavelets calculated; J = 1")
                return (wavelets.flatten(wavelet_dictionary, wavelet_sizes))
    
            Psi_j_tilde = wavelets.normalize(Psi_j_tilde)
            wavelet_sizes.append(Psi_j_tilde.shape[1])
            wavelet_dictionary += [Psi_j_tilde]
    
            for i in tqdm(range(2,self.wavelet_J), disable=self.verbose==False):
                P_j_new = np.linalg.matrix_power(P_j,self.wavelet_power)
                Psi_j = P_j - P_j_new
                P_j = P_j_new
                Psi_j_tilde = wavelets.column_subset(Psi_j, epsilon=self.qr_epsilon)
                if Psi_j_tilde.shape[1] == 0: 
                    _logger.log_info("Wavelets calculated; J = %s" %i)
                    return (wavelets.flatten(wavelet_dictionary, wavelet_sizes))
    
                Psi_j_tilde = wavelets.normalize(Psi_j_tilde)
    
                wavelet_sizes.append(Psi_j_tilde.shape[1])
                wavelet_dictionary += [Psi_j_tilde]
        else:
            _logger.log_info("Calculating Wavelets J = %s" % self.wavelet_J)
            wavelet_dictionary += [I-P_j]
            wavelet_sizes.append((I-P_j).shape[1])
            for i in tqdm(range(2,self.wavelet_J), disable=self.verbose==False):
                P_j_new = np.linalg.matrix_power(P_j,self.wavelet_power)
                Psi_j = P_j - P_j_new
                P_j = P_j_new
                Psi_j = wavelets.normalize(Psi_j)
                wavelet_sizes.append(Psi_j.shape[1])
                wavelet_dictionary += [Psi_j]
                
        self.wavelet_dictionary, self.wavelet_sizes = wavelets.flatten(wavelet_dictionary, wavelet_sizes)

    def get_gene_embeddings(self, signals):
        """Get gene features embedded in principle component space and autoencoded space.

        Parameters
        ----------
        signals: array-like, shape=[n_features, n_samples]
            Input signals defined on nodes of cell-cell graph. Accepted data types: `numpy.ndarray`, `pd.DataFrame`.

        Returns
        ----------
        signals_ae: array, shape=[n_features, embedding_dim]
            Signals embedded with autoencoder into `embedding_dim`-dimensional space.
        signals_pc: array, shape=[n_features, pc_dim]
            Signals embedded with PCA into `pc_dim`-dimensional space.
        """
        
        if self.wavelet_dictionary is None:
            raise ValueError('Run gspa_op.build_wavelet_dictionary')

        if self.condensation_groupings is not None:
            signals = graphs.aggregate_signals_over_condensed_nodes(signals.T, self.condensation_groupings).T
        
        self.signals_projected = embedding.project(signals, self.wavelet_dictionary)
        signals_pc = embedding.svd(self.signals_projected, n_components=self.pc_dim)
        signals_ae = embedding.run_ae(signals_pc, random_state=self.random_state, act=self.activation, bias=self.bias,
                            dim=self.embedding_dim, num_layers=self.num_layers, dropout=self.dropout, lr=self.lr,
                            epochs=self.epochs, val_prop=self.val_prop, weight_decay=self.weight_decay, patience=self.patience)
        
        return (signals_ae, signals_pc)

    def calculate_localization(self, signals=None):
        """Calculates localization for signals.

        Parameters
        ----------
        signals: array-like, optional, shape=[n_features, n_samples]
            Input signals defined on nodes of cell-cell graph. Accepted data types: `numpy.ndarray`, `pd.DataFrame`. If signals is None, calculates localization for gene signals inputted to `get_gene_embeddings`.

        Returns
        ----------
        localization_score: array, shape=[n_features,]
            Localization score for each gene, where higher score indicates the gene is more localized on the cell-cell graph.
        """
        
        if self.wavelet_dictionary is None:
            raise ValueError('Run gspa_op.build_wavelet_dictionary')
        if signals is not None:
            _logger.log_info(f"Computing localization with provided signals.")
            if self.condensation_groupings is not None:
                signals = graphs.aggregate_signals_over_condensed_nodes(signals.T, self.condensation_groupings).T
            signals_projected = embedding.project(signals, self.wavelet_dictionary)
            uniform_signal = np.ones((1, self.wavelet_dictionary.shape[0]))
            uniform_projected = embedding.project(uniform_signal, self.wavelet_dictionary)
            localization_score = spatial.distance.cdist(uniform_projected, signals_projected).reshape(-1,)

        else:
            if self.signals_projected is None:
                raise ValueError('Provide signals to map to dictionary or run gspa_op.get_gene_embeddings')
            else:
                _logger.log_info(f"Computing localization with signals used for gene embedding.")
                uniform_signal = np.ones((1, self.wavelet_dictionary.shape[0]))
                uniform_projected = embedding.project(uniform_signal, self.wavelet_dictionary)
                localization_score = spatial.distance.cdist(uniform_projected, self.signals_projected).reshape(-1,)
                
        return (localization_score)
            
    def calculate_cell_type_specificity(self, cell_type_assignments, cell_type, signals=None):
        """Calculates cell type specificity for each signal to provided cell type of interest.

        Parameters
        ----------
        cell_type_assignments: array-like, shape=[n_samples,] 
            Cluster or cell type assignments to cell nodes.
        cell_type: string
            Cluster name or cell type of interest.
        signals: array-like, optional, shape=[n_features, n_samples]
            Input signals defined on nodes of cell-cell graph. Accepted data types: `numpy.ndarray`, `pd.DataFrame`. If signals is None, calculates localization for gene signals inputted to `get_gene_embeddings`.

        Returns
        ----------
        specificity_score: array, shape=[n_features,]
            Cell type specificity score for each gene, where higher score indicates the gene is more specific to provided cell type.
        """
        
        cell_type_assignments = np.array(cell_type_assignments)
        if cell_type not in cell_type_assignments:
            raise ValueError('Cell type not found in cell type assignments')
        if self.wavelet_dictionary is None:
            raise ValueError('Run gspa_op.build_wavelet_dictionary')

        if signals is not None:
            _logger.log_info(f"Computing cell type specificity with provided signals.")
            if self.condensation_groupings is not None:
                signals = graphs.aggregate_signals_over_condensed_nodes(signals.T, self.condensation_groupings).T
            signals_projected = embedding.project(signals, self.wavelet_dictionary)
            
            cell_type_signal = (cell_type_assignments == cell_type).astype(int).reshape(1, -1)
            if self.condensation_groupings is not None:
                cell_type_signal = graphs.aggregate_signals_over_condensed_nodes(cell_type_signal.T, self.condensation_groupings).T
            cell_type_projected = embedding.project(cell_type_signal, self.wavelet_dictionary)
            specificity_score = -1*spatial.distance.cdist(cell_type_projected, signals_projected).reshape(-1,)

        else:
            if self.signals_projected is None:
                raise ValueError('Provide signals to map to dictionary or run gspa_op.get_gene_embeddings')
            else:
                _logger.log_info(f"Computing cell type specificity with signals used for gene embedding")
                cell_type_signal = (cell_type_assignments == cell_type).astype(int).reshape(1, -1)
                if self.condensation_groupings is not None:
                    cell_type_signal = graphs.aggregate_signals_over_condensed_nodes(cell_type_signal.T, self.condensation_groupings).T
                cell_type_projected = embedding.project(cell_type_signal, self.wavelet_dictionary)
                specificity_score = -1*spatial.distance.cdist(cell_type_projected, self.signals_projected).reshape(-1,)
        
        return (specificity_score)
