import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from avcv.all import *
from FOD.dataset import AutoFocusDataset
from FOD.Trainer import Trainer
from ple.all import *
from torch.utils.data import ConcatDataset, DataLoader

parser = argparse.ArgumentParser()


parser.add_argument('--ckpt', default=None)
parser.add_argument('--config', default='training/config.json')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


with open(args.config, 'r') as f:
    config = json.load(f)




if args.debug:
    config['General']['batch_size'] = 2
    config['General']['num_epochs'] = 1
    config['General']['num_workers'] = 0
    config['General']['seed'] = 0




np.random.seed(config['General']['seed'])

list_data = config['Dataset']['paths']['list_datasets']


autofocus_datasets_train = []
for dataset_name in list_data:
    autofocus_datasets_train.append(
        AutoFocusDataset(config, dataset_name, 'train'))
train_data = ConcatDataset(autofocus_datasets_train)
train_dataloader = DataLoader(
    train_data, batch_size=config['General']['batch_size'], shuffle=True, num_workers=config['General']['num_workers'])
batch = next(iter(train_dataloader))
print(f'{batch.keys()=}')
# validation set
autofocus_datasets_val = []
for dataset_name in list_data:
    autofocus_datasets_val.append(
        AutoFocusDataset(config, dataset_name, 'val'))
val_data = ConcatDataset(autofocus_datasets_val)
val_loader = DataLoader(
    val_data, batch_size=config['General']['batch_size'], shuffle=False, num_workers=config['General']['num_workers'])

model_trainer = Trainer(config)

loss_depth = model_trainer.loss_depth
loss_seg = model_trainer.loss_segmentation
model = model_trainer.model


class CustomLitModel(LitModel):
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['depth']
        depth_pred, seg_pred = self(x)
        ld = loss_depth(depth_pred, y)
        ls = loss_seg(seg_pred.permute(
            [0, 2, 3, 1]).reshape(-1, 3), batch['segmentation'].permute([0, 2, 3, 1]).reshape(-1))
        loss = ld  # + ls

        self.log('train/loss', ld, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/loss_seg', ls, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['depth']
        depth_pred, seg_pred = self(x)
        ld = loss_depth(depth_pred, y)
        ls = loss_seg(seg_pred.permute(
            [0, 2, 3, 1]).reshape(-1, 3), batch['segmentation'].permute([0, 2, 3, 1]).reshape(-1))
        self.log('val/loss', ld, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/loss_seg', ls, on_step=False,
                 on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x = batch['image']
        depth_pred, seg_pred = self(x)
        for i in range(len(depth_pred)):
            depth = depth_pred[i].detach().cpu().numpy()[0]
            filename = batch['filename'][i]
            depth = (depth*3000).astype(np.uint16)
            conf = seg_pred[i].softmax(0)[2].detach().cpu().numpy()
            conf = (conf*255).astype(np.uint8)
            mmcv.imwrite(depth, f'./output/depth/{filename}.png')
            mmcv.imwrite(conf, f'./output/segmentation/{filename}.png')


sched = fn_schedule_cosine_with_warmpup_decay_timm(
    num_epochs=config['General']['epochs'],
    num_steps_per_epoch=len(train_dataloader)//config['General']['gpus'],
    num_epochs_per_cycle=config['General']['epochs']//3,
    min_lr=1/100,
    cycle_decay=0.7,
)
def optim(params): return torch.optim.Adam(
    params, lr=config['General']['lr_scratch'])


lit_model = CustomLitModel(
    model.cpu(), create_lr_scheduler_fn=sched, create_optimizer_fn=optim)

if args.ckpt is not None:
    trainer = get_trainer('focus_on_depth/predict', gpus=1, strategy='dp')
    ckpt = torch.load(args.ckpt)['state_dict']
    lit_model.load_state_dict(ckpt)
    pred_data = AutoFocusDataset(config, dataset_name, 'predict')
    T = pred_data.transform_image

    class DS:
        def __init__(self):
            self.paths = paths = glob(
                '/home/anhvth8/gitprojects/IDepthDataset/data/221222-1/rgbd/*.jpg')

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            x = T(Image.open(self.paths[idx]))
            return dict(image=x, filename=self.paths[idx].split('/')[-1].split('.')[0])

    pred_loader = DataLoader(
        DS(), batch_size=config['General']['batch_size'], shuffle=False, num_workers=4)
    batch = next(iter(pred_loader))
    trainer.predict(lit_model, pred_loader)
else:
    trainer = get_trainer('focus_on_depth',
                          config['General']['epochs'], gpus=config['General']['gpus'], strategy=config['General']['strategy'],
                          accelerator="cpu" if not torch.cuda.is_available() else "gpu",
                          refresh_rate=1 if args.debug else 5,
                          overfit_batches=2 if args.debug else 0)
    logger.info('Training from scratch')
    trainer.fit(lit_model, train_dataloader, val_loader)
