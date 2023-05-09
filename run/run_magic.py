import json
import os
import magic
import scprep
import numpy as np

def run_magic(G, args):
    
    magic_op = magic.MAGIC(t=args.diffusion_t)
    magic_op.graph = G
        
    return (embedding)