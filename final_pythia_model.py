"""Inference model to be checked."""
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from pythia.tasks.processors import VocabProcessor, VQAAnswerProcessor
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from pythia.common.sample import Sample, SampleList
from pythia.utils.configuration import ConfigNode
from pythia.common.registry import registry
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from pythia.models.pythia import Pythia
import torchvision.models as models
import torch.nn.functional as F
import torch
import yaml

torch.backends.cudnn.enabled = False

class PythiaVQA(torch.nn.Module):

    def __init__(self, device):
        super(PythiaVQA, self).__init__()
        self.device = device
        self._init_processors()
        self.pythia_model = self._build_pythia_model()
        self.detection_model = self._build_detection_model()
        self.resnet152_model = self._build_resnet_model()

    def _init_processors(self):
        with open("pythia.yml") as f:
            config = yaml.load(f)

        config = ConfigNode(config)
        config.training_parameters.evalai_inference = True
        registry.register("config", config)

        self.config = config

        vqa_config = config.task_attributes.vqa.dataset_attributes.vqa2
        text_processor_config = vqa_config.processors.text_processor
        answer_processor_config = vqa_config.processors.answer_processor

        text_processor_config.params.vocab.vocab_file = "data/vocabulary_100k.txt"
        answer_processor_config.params.vocab_file = "data/answers_vqa.txt"
        # Add preprocessor as that will needed when we are getting questions from user
        self.text_processor = VocabProcessor(text_processor_config.params)
        self.answer_processor = VQAAnswerProcessor(answer_processor_config.params)

        registry.register("vqa2_text_processor", self.text_processor)
        registry.register("vqa2_answer_processor", self.answer_processor)
        registry.register("vqa2_num_final_outputs",
                          self.answer_processor.get_vocab_size())

    def _build_pythia_model(self):
        # state_dict = torch.load('models/pythia.pth')    # pythia
        # state_dict = torch.load('models/squint.ckpt')['model']    # squint
        state_dict = torch.load('models/sort.ckpt')['model']    # sort

        model_config = self.config.model_attributes.pythia
        model = Pythia(model_config)
        model.build()
        model.init_losses_and_metrics()

        if list(state_dict.keys())[0].startswith('module') and \
                not hasattr(model, 'module'):
            state_dict = self._multi_gpu_state_to_single(state_dict)

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _build_resnet_model(self):
        resnet152 = models.resnet152(pretrained=True)
        resnet152.eval()
        modules = list(resnet152.children())[:-2]

        resnet152_model = torch.nn.Sequential(*modules)
        resnet152_model.to(self.device)
        resnet152_model.eval()
        return resnet152_model

    def _multi_gpu_state_to_single(self, state_dict):
        new_sd = {}
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                raise TypeError("Not a multiple GPU state of dict")
            k1 = k[7:]
            new_sd[k1] = v
        return new_sd

    def _build_detection_model(self):

        cfg.merge_from_file('configs/detectron_model.yaml')
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load('models/detectron_model.pth',
                                map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to(self.device)
        model.eval()
        return model

    def _process_feature_extraction(self, output,
                                    im_scales,
                                    feat_name='fc6',
                                    conf_thresh=0.2):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feat_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]

            max_conf = torch.zeros((scores.shape[0])).to(cur_device)

            for cls_ind in range(1, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                             cls_scores[keep],
                                             max_conf[keep])

            keep_boxes = torch.argsort(max_conf, descending=True)[:100]
            feat_list.append(feats[i][keep_boxes])
        return feat_list

    def masked_unk_softmax(self, x, dim, mask_idx):
        x1 = F.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def get_resnet_features(self, img):
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        img = img.unsqueeze(0).to(self.device)

        features = self.resnet152_model.forward(img).permute(0, 2, 3, 1)
        features = features.view(196, 2048)
        return features

    def get_detectron_features(self, im, im_scale):
        current_img_list = to_image_list(im, size_divisible=32)
        current_img_list = current_img_list.to(self.device)
        with torch.no_grad():
            output = self.detection_model.forward(current_img_list)
        feat_list = self._process_feature_extraction(output, im_scale,
                                                     'fc6', 0.2)
        return feat_list[0]

    def forward(self, batch):
        question = batch['question'][0]
        detectron_img = batch['detectron_img'].squeeze()
        detectron_scale = [batch['detectron_scale'].item()]
        resnet_img = batch['resnet_img'].squeeze()

        detectron_features = self.get_detectron_features(detectron_img, detectron_scale)
        resnet_features = self.get_resnet_features(resnet_img)
        sample = Sample()

        processed_text = self.text_processor({"text": question})
        sample.text = processed_text["text"]
        sample.text_len = len(processed_text["tokens"])

        sample.image_feature_0 = detectron_features
        sample.image_info_0 = Sample({
            "max_features": torch.tensor(100, dtype=torch.long)
        })

        sample.image_feature_1 = resnet_features

        sample_list = SampleList([sample])
        sample_list = sample_list.to(self.device)

        scores = self.pythia_model.forward(sample_list)["scores"]

        return scores
