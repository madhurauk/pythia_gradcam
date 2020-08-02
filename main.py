from pythia_grad_cam import GradCAM
from pythia_grad_cam import _BaseWrapper
from pythia_model_2 import PythiaVQA
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

myGradCAM = GradCAM(_BaseWrapper(PythiaVQA(device)))