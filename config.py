import argparse

config_args = {
    'training_config': {
        'lr': (0.01, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'epochs': (100, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'patience': (50, 'patience for early stopping'),
        'min-epochs': (50, 'do not early stop before min-epochs'),
        'seed': (1234, 'seed for training'),
        'walk-length': (80, 'walk length for Node2Vec'),
        'num-walks': (10, 'number of walks for Node2Vec'),
        'task': ('localization', 'task for paper, can be any of [coexpression, localization]'),
    },
    'model_config': {
        'model': ('GSPA', 'which encoder to use, can be any of [Node2Vec, GAE, GSPA]'),
        'dim': (16, 'embedding dimension'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use [of relu, complexrelu (MagNet), tanh, or None]'),
        'num-classes': (2, 'magnet link prediction class number'),
        'attention': (True, 'use graph attention layers for graph convolutions')
     },
    'data_config': {
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
        'save-as': ('0', 'name for embedding iteration'),
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
