import importlib
import helpers
import pipelines
import pipeline_helpers
import generate_data
import numpy as np
import random
import torch


def reload_all():
  importlib.reload(helpers)
  importlib.reload(pipelines)
  importlib.reload(pipeline_helpers)
  importlib.reload(generate_data)


def generate_seed():
  np.random.seed(42)
  random.seed(42)
  torch.manual_seed(42)
  
def reload_notebook_cell():
  reload_all()
  generate_seed()
  
  
  
  