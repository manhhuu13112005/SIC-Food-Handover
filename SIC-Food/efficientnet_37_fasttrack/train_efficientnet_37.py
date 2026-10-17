from pathlib import Path
import argparse, json, hashlib, os, csv, time
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image, ImageOps
from tqdm.auto import tqdm

class Images(Dataset):
    def __init__(self, frame, names, tfm):
        self.paths = frame.path.astype(str).tolist()
        ids = {name:i for i,name in enumerate(names)}
        self.labels = [ids[n] for n in frame.target_class]
        self.tfm = tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tfm(ImageOps.exif_transpose(im).convert('RGB'))
        return x, self.labels[i]

def signature(df, names):
    h = hashlib.sha256(json.dumps(names).encode())
    for p, name in zip(df.path, df.target_class):
        h.update(f'{p}\0{name}\n'.encode('utf-8'))
    return h.hexdigest()[:16]

def cached_features(frame, tag, names, model, tfm, cache_root, batch):
    sig = signature(frame, names)
    cache = cache_root / f'{tag}_{sig}'
    cache.mkdir(parents=True, exist_ok=True)
    ds = Images(frame, names, tfm)
    chunk_size = 512
    for k in tqdm(range((len(ds)+chunk_size-1)//chunk_size), desc=f'Cache {tag}'):
        a, b = k*chunk_size, min((k+1)*chunk_size, len(ds))
        target = cache / f'{k:05d}.pt'
        if target.exists():
            try:
                chk = torch.load(target, weights_only=True, map_location='cpu')
                if chk['x'].shape == (b-a, 1280) and chk['y'].shape == (b-a,):
                    continue
            except Exception:
                pass
        loader = DataLoader(Subset(ds, range(a,b)), batch_size=batch, num_workers=0)
        xx, yy = [], []
        with torch.inference_mode():
            for x, y in loader:
                features = model.features(x)
                features = torch.flatten(model.avgpool(features), 1)
                xx.append(features.cpu()); yy.append(y.cpu())
        temp = cache / f'{k:05d}.tmp'
        torch.save({'x':torch.cat(xx), 'y':torch.cat(yy)}, temp)
        os.replace(temp, target)
    xx, yy = [], []
    for fp in sorted(cache.glob('*.pt')):
        obj = torch.load(fp, weights_only=True, map_location='cpu')
        xx.append(obj['x']); yy.append(obj['y'])
    return torch.cat(xx), torch.cat(yy), sig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'C:\Users\ASUS\SIC-Food')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--image-batch-size', type=int, default=16)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(a.threads)
    root = Path(a.root)
    source = root/'datasets'/'efficientnet_pilot_37'
    output = root/'models'/'efficientnet_37_fasttrack'
    output.mkdir(parents=True,exist_ok=True)
    with (source/'class_names.json').open(encoding='utf-8') as f:
        names = json.load(f)
    assert len(names)==37 and len(set(names))==37
    train = pd.read_csv(source/'train_manifest.csv').drop_duplicates('path').reset_index(drop=True)
    val_all = pd.read_csv(source/'v3_eval_reference.csv')
    val = val_all[val_all['split']=='val'].drop_duplicates('path').reset_index(drop=True)
    assert set(train.path).isdisjoint(set(val.path))
    assert set(train.target_class).issubset(set(names))
    print(f'Train candidates: {len(train)}, validation V3 (26 classes): {len(val)}')
    print('CAUTION: prototype; V4/cabbage labels unapproved, no independent 37-class validation.')
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    net = efficientnet_b0(weights=weights).eval()
    for p in net.features.parameters(): p.requires_grad_(False)
    tfm = weights.transforms()
    print('Preprocessing:', tfm)
    cx = output/'feature_cache'
    train_x, train_y, sig_t = cached_features(train,'train',names,net,tfm,cx,a.image_batch_size)
    val_x, val_y, sig_v = cached_features(val,'val_v3',names,net,tfm,cx,a.image_batch_size)
    counts = torch.bincount(train_y,minlength=len(names)).float()
    assert (counts>0).all()
    weights_loss = torch.sqrt(counts.mean()/counts).clamp(0.25,4.0)
    criterion = nn.CrossEntropyLoss(weight=weights_loss)
    head = nn.Linear(1280,len(names))
    opt = torch.optim.AdamW(head.parameters(),lr=0.002,weight_decay=1e-4)
    state_path = output/'training_state.pt'
    model_path = output/'efficientnet_b0_37_prototype.pth'
    history_path = output/'training_history.csv'
    signature_both = [sig_t,sig_v]
    history=[]; start=1
    if state_path.exists():
        state=torch.load(state_path,weights_only=True,map_location='cpu')
        if state['signature'] != signature_both or state['class_names']!=names:
            raise RuntimeError('Training state differs from manifests: move old training_state.pt first.')
        head.load_state_dict(state['head']); opt.load_state_dict(state['optimizer'])
        history=state['history']; start=state['epoch']+1
        print('Resume at epoch',start)
    loader=DataLoader(TensorDataset(train_x,train_y),batch_size=256,shuffle=True,num_workers=0)
    val_groups=sorted(set(val_y.tolist()))
    for epoch in range(start,a.epochs+1):
        ts=time.time(); head.train(); n=0; correct=0; loss_sum=0
        for x,y in loader:
            opt.zero_grad(set_to_none=True)
            out=head(x); loss=criterion(out,y)
            loss.backward(); opt.step()
            n+=len(y); correct+=(out.argmax(1)==y).sum().item(); loss_sum+=loss.item()*len(y)
        head.eval()
        with torch.inference_mode():
            pred=head(val_x).argmax(1)
            val_acc=(pred==val_y).float().mean().item()
            macro=sum((pred[val_y==c]==c).float().mean().item() for c in val_groups)/len(val_groups)
        record={'epoch':epoch,'train_loss':round(loss_sum/n,5),'train_acc':round(correct/n*100,2),
                'v3_val_26_acc_37_output':round(val_acc*100,2),
                'v3_val_26_macro_recall':round(macro*100,2),'seconds':round(time.time()-ts,2)}
        history.append(record); print(record,flush=True)
        net.classifier[1]=nn.Linear(1280,len(names))
        net.classifier[1].load_state_dict(head.state_dict())
        temporary=model_path.with_suffix('.tmp')
        torch.save({'model_state_dict':net.state_dict(),'class_names':names,'num_classes':37,
                    'epoch':epoch,'pretrained_weights':weights.name,
                    'training_type':'prototype_frozen_backbone',
                    'validation_scope':'V3 val only 26/37 classes'},temporary)
        os.replace(temporary,model_path)
        temporary=state_path.with_suffix('.tmp')
        torch.save({'head':head.state_dict(),'optimizer':opt.state_dict(),'class_names':names,
                    'signature':signature_both,'epoch':epoch,'history':history},temporary)
        os.replace(temporary,state_path)
        with history_path.open('w',newline='',encoding='utf-8-sig') as file:
            w=csv.DictWriter(file,fieldnames=history[0].keys()); w.writeheader(); w.writerows(history)
    print('Model:',model_path)
    print('History:',history_path)
    print('V3 TEST NOT USED. New 11 classes have no independent validation!')

if __name__=='__main__': main()
