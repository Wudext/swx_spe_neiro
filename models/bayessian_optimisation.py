import optuna
import torch
import numpy as np
from sklearn.model_selection import KFold
from models.neural import SolarFlareNet, load_and_preprocess_data  # Предполагаемая структура проекта
