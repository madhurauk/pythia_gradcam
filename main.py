from pythia_grad_cam import GradCAM
#from pythia_grad_cam import _BaseWrapper
from pythia_model_2 import PythiaVQA
import torch
import sys
import cv2
#sys.path.append('/srv/share3/mummettuguli3/code/pythia_gradcam/vqa-maskrcnn-benchmark')
#print("one");
#print(sys.path);
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img = cv2.imread('images/cat_dog.jpg', 1)
img = cv2.resize(img, (224, 224))
img = torch.as_tensor(img)
myGradCAM = GradCAM(PythiaVQA(device))

myInput = {
'question':[
'What animal?'],
'detectron_img':img,
'detectron_scale':torch.tensor(224),
'resnet_img':img
}
#myGradCAM.forward(img, (224,224))
scores = myGradCAM.forward(myInput, (224,224))
print("scores ",scores)
