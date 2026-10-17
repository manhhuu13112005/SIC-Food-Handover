"""Simple EfficientNet 37-class predictor for an existing website backend."""
from pathlib import Path
import torch
from torch import nn
from PIL import Image, ImageOps
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNet37Predictor:
    def __init__(self, checkpoint_path):
        ckpt = torch.load(Path(checkpoint_path), map_location='cpu', weights_only=True)
        self.names = list(ckpt['class_names'])
        if len(self.names) != 37 or ckpt.get('pretrained_weights') != 'IMAGENET1K_V1':
            raise ValueError('Checkpoint is not the expected EfficientNet-B0 37-class fast-track model.')
        self.net = efficientnet_b0(weights=None)
        self.net.classifier[1] = nn.Linear(self.net.classifier[1].in_features,len(self.names))
        self.net.load_state_dict(ckpt['model_state_dict'], strict=True)
        self.net.eval()
        self.transform = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    def predict(self, image: Image.Image, top_k=3):
        image=ImageOps.exif_transpose(image).convert('RGB')
        x=self.transform(image).unsqueeze(0)
        with torch.inference_mode():
            proba=self.net(x).softmax(1).squeeze(0)
            scores, indexes=proba.topk(min(max(1,top_k),len(self.names)))
        result=[]
        for confidence, idx in zip(scores.tolist(), indexes.tolist()):
            label=self.names[idx]
            ingredient, state=label.split('__',1)
            result.append({'label':label,'ingredient':ingredient,
                           'appearance':state,'confidence':round(confidence,6)})
        return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--image',required=True)
    args=p.parse_args()
    classifier=EfficientNet37Predictor(args.model)
    with Image.open(args.image) as image:
        for prediction in classifier.predict(image): print(prediction)
