import argparse

config_args = {
    'training_config': {
        'lr': (0.01, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'epochs': (200, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'patience': (50, 'patience for early stopping'),
        'min-epochs': (50, 'do not early stop before min-epochs'),
        'seed': (1234, 'seed for training'),
        'c': (1.0, 'hyperbolic radius'),
        'walk-length': (80, 'walk length for Node2Vec'),
        'num-walks': (10, 'number of walks for Node2Vec'),
        'task': ('existence', 'task for MagNet, can be any of [existence, direction, three_class_digraph]'),
        'q': (0.2, 'charge parameter for directed methods'),
        'edge_attribute': (0, 'include edge attribute for KGE models'),
        'margin': (9.0, 'margin for NSSA loss for KGE models'),
        'temperature': (1.0, 'adversarial temperature for NSSA loss for KGE models'),
        'compute-scattering': (False, 'compute scattering from data versus use pretrained scattering coefficients'),
    },
    'model_config': {
        'model': ('Node2Vec', 'which encoder to use, can be any of [Node2Vec, TransE, ConvE, PoincareMap, GAE, HGCN, MagNet, UDS-AE, DS-AE]'),
        'dim': (16, 'embedding dimension'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use [of relu, complexrelu (MagNet), tanh, or None]'),
        'num-classes': (2, 'magnet link prediction class number'),
     },
    'data_config': {
        'dataset': ('omnipath', 'which dataset to use, can be between [omnipath, lrtree]'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
        'save-as': ('0', 'name for embedding iteration'),
        'symmetrize-adj': (True, 'symmetrize adjacency matrix for undirected graphs')
    }
}

def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
