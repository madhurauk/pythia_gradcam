"""Main file. Visualizing predicted answer"""
from torch.utils.data import DataLoader
from pythia_dataset import VQA_Dataset
from tools import save_attention_map
import matplotlib.pyplot as plt
import pythia_grad_cam as pgc
import numpy as np
import pickle
import torch
import time
import pdb
import cv2
import sys
import gc

DEVICE = "cuda:0"

TARGET_IMAGE_SIZE = [448, 448]
CHANNEL_MEAN = [0.485, 0.456, 0.406]
#CHANNEL_MEAN = (0.485,0.456,0.406)
CHANNEL_STD = [0.229, 0.224, 0.225]
#CHANNEL_STD = (0.229, 0.224, 0.225)
IMAGE_SHAPE = torch.Size(TARGET_IMAGE_SIZE)

use_pythia = True
dataset_path = '/coc/scratch/mummettuguli3/data/vqa2.0/working/'
# Validation Set
dataType = "val"
dataSubType = dataType + '2014'
annFile = dataset_path + 'Annotations/v2_mscoco_val2014_annotations.json'
quesFile = dataset_path + 'Questions/v2_OpenEnded_mscoco_val2014_questions.json'
imgDir = dataset_path + 'Images/val2014/'
#evalIds = "evaluated_val_ids.npy"
evalIds = 'valid.npy'

if use_pythia:
    from pythia_model_2 import PythiaVQA as Model
    model_dir = "models"
else:
    from ban_model import BanVQA as Model
    model_dir = "ban"

def print_progress(start, j, dataset_len):
    progress = ((j + 1) / dataset_len) * 100
    elapsed = time.time() - start
    time_per_annotation = elapsed / (j + 1)

    finished_in = time_per_annotation * (dataset_len - (j + 1))
    day = finished_in // (24 * 3600)
    finished_in = finished_in % (24 * 3600)
    hour = finished_in // 3600
    finished_in %= 3600
    minutes = finished_in // 60
    finished_in %= 60
    seconds = finished_in
    print("Iteration: {} | Progress: {}% | Finished in: {}d {}h {}m {}s | Time Per Annotation: {}s".format(j, round(
        progress, 6), round(day), round(hour), round(minutes), round(seconds), round(time_per_annotation, 2)))

def predict():
    splits = 1
    subset = -1
    checkpoint = 0
    vqa_dataset = VQA_Dataset(evalIds, annFile, quesFile, imgDir, dataSubType, TARGET_IMAGE_SIZE, CHANNEL_MEAN, CHANNEL_STD, splits=splits, subset=subset, checkpoint=checkpoint)
    dataset_len = vqa_dataset.__len__()
    print("subset: {}".format(subset))
    print("checkpoint: {}".format(checkpoint))
    print("vqa_dataset size: {}".format(dataset_len))
    with torch.enable_grad():
        layer = 'resnet152_model.7'
        vqa_model = Model(DEVICE)
        vqa_model.eval()
        vqa_model_GCAM = pgc.GradCAM(model=vqa_model, candidate_layers=[layer])
        data_loader = DataLoader(vqa_dataset, batch_size=1, shuffle=False) #remove 
        start = time.time()
        results = []
        result_index = 0
        answer_dir = model_dir + "/" + model_dir + "_answers_" + dataType + "/" + model_dir + "_pred_subset_"
        questions, answs = {}, {}
        count_loop = 0
        for j, batch in enumerate(data_loader): #question = string , image = path
            if count_loop == 750:
                break
            print_progress(start, checkpoint+j, dataset_len)
            #print("in main_dataloader batch: ",batch)
            annId = batch['annId'].item()
            question = batch['question'][0]
            raw_image = batch['raw_image'].squeeze() #cv2.imread
            raw_image = raw_image.cpu().numpy()
            raw_image = cv2.resize(raw_image, tuple(TARGET_IMAGE_SIZE))

            print("QID: ", annId)
            #print("batch: ", batch)
            count_loop = count_loop+1

            actual, indices = vqa_model_GCAM.forward(batch, IMAGE_SHAPE)
            top_indices = indices[0]
            top_scores = actual[0]

            probs = []
            answers = []

            for idx, score in enumerate(top_scores):
                probs.append(score.item())
                answers.append(
                    vqa_model.answer_processor.idx2word(top_indices[idx].item())
                )

            # get questions and answers for visualization later
            answs[annId] = answers[0]
            questions[annId] = question

            results.append([annId, answers[0], probs[0]])

            vqa_model_GCAM.backward(ids=indices[:, [0]])
            attention_map_GradCAM = vqa_model_GCAM.generate(target_layer=layer)

            attention_map_GradCAM = attention_map_GradCAM.squeeze().cpu().numpy()

            save_attention_map(attn_map=attention_map_GradCAM, qid=annId)

        # save questions and answs
        with open('questions.pickle', 'wb') as handle:
            pickle.dump(questions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('answers.pickle', 'wb') as handle:
            pickle.dump(answs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def print_layer_names(model, full=False):
    if not full:
        print(list(model.named_modules())[0])
    else:
        print(*list(model.named_modules()), sep='\n')


if __name__ == "__main__":
    predict()
