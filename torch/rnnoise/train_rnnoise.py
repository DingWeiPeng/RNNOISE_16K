"""
/* Copyright (c) 2024 Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import os
import rnnoise
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('features', type=str, help='path to feature file in .f32 format')
parser.add_argument('output', type=str, help='path to output folder')

parser.add_argument('--nb-bands', type=int, help="NB_BANDS in denoise.h", default=24) # 48k, 32; 16k, 24
parser.add_argument('--suffix', type=str, help="model name suffix", default="")
parser.add_argument('--cuda-visible-devices', type=str, help="comma separates list of cuda visible device indices, default: CUDA_VISIBLE_DEVICES", default=None)


model_group = parser.add_argument_group(title="model parameters")
model_group.add_argument('--cond-size', type=int, help="first conditioning size, default: 128", default=128)
model_group.add_argument('--gru-size', type=int, help="first conditioning size, default: 384", default=384)

training_group = parser.add_argument_group(title="training parameters")
training_group.add_argument('--batch-size', type=int, help="batch size, default: 128", default=80)
training_group.add_argument('--lr', type=float, help='learning rate, default: 1e-3', default=1e-3)
training_group.add_argument('--epochs', type=int, help='number of training epochs, default: 200', default=200)
training_group.add_argument('--sequence-length', type=int, help='sequence length, default: 1000', default=2000) # 由2000变成了2000
training_group.add_argument('--lr-decay', type=float, help='learning rate decay factor, default: 5e-5', default=5e-5)
training_group.add_argument('--initial-checkpoint', type=str, help='initial checkpoint to start training from, default: None', default=None)
training_group.add_argument('--gamma', type=float, help='perceptual exponent (default 0.25)', default=0.25)
training_group.add_argument('--sparse', action='store_true')

args = parser.parse_args()



class RNNoiseDataset(torch.utils.data.Dataset):
    def __init__(self,
                features_file,
                nb_bands,
                sequence_length=1000): #序列长度我把它从2000改为了1000
        # rnnoise中划分的频带数目。48k为32,16k为24
        self.nb_bands = nb_bands
        self.sequence_length = sequence_length

        self.data = np.memmap(features_file, dtype='float32', mode='r')
        # 一帧音频生成的特征总数
        self.dim = 3*self.nb_bands+2

        self.nb_sequences = self.data.shape[0]//self.sequence_length//self.dim
        self.data = self.data[:self.nb_sequences*self.sequence_length*self.dim]

        # self.sequence_length为多少帧，一帧长度为10ms。dim为一帧特征的长度。self.nb_sequences为多少个多少帧。
        self.data = np.reshape(self.data, (self.nb_sequences, self.sequence_length, self.dim))

        # 一帧音频生成的特征
        self.features_num = 2*self.nb_bands+1
        self.gain_num = self.nb_bands
        self.vad_num = 1

    def __len__(self):
        return self.nb_sequences

    def __getitem__(self, index):
        # 对应着features, gain, vad
        return self.data[index, :, :self.features_num].copy(), self.data[index, :, self.features_num:self.dim-1].copy(), self.data[index, :, -1:].copy()

def mask(g):
    return torch.clamp(g+1, max=1)

adam_betas = [0.8, 0.98]
adam_eps = 1e-8
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
sequence_length = args.sequence_length
lr_decay = args.lr_decay

cond_size  = args.cond_size
gru_size  = args.gru_size
nb_bands = args.nb_bands
checkpoint_dir = os.path.join(args.output, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = dict()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

checkpoint['model_args']    = ()
checkpoint['model_kwargs']  = {'cond_size': cond_size, 'gru_size': gru_size, 'nb_bands':nb_bands}
model = rnnoise.RNNoise(*checkpoint['model_args'], **checkpoint['model_kwargs'])

if type(args.initial_checkpoint) != type(None):
    checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=False)

checkpoint['state_dict']    = model.state_dict()

dataset = RNNoiseDataset(features_file=args.features, nb_bands=args.nb_bands, sequence_length=args.sequence_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)


optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps)


# learning rate scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda x : 1 / (1 + lr_decay * x))

gamma = args.gamma

# 为了针对16k采样率做修改，我们需要修改一些参数
# 在dump_features里，写入特征的函数如下：      
# fwrite(features, sizeof(float), NB_FEATURES, fout); 
# fwrite(g, sizeof(float), NB_BANDS, fout);
# fwrite(&vad_target, sizeof(float), 1, fout);
# 其中，NB_BANDS = 24, NB_FEATURES = 2*NB_BANDS+1
if __name__ == '__main__':
    model.to(device)
    states = None
    for epoch in range(1, epochs + 1):

        running_gain_loss = 0
        running_vad_loss = 0
        running_loss = 0

        print(f"training epoch {epoch}...")
        with tqdm.tqdm(dataloader, unit='batch') as tepoch:
            for i, (features, gain, vad) in enumerate(tepoch):
                optimizer.zero_grad()
                features = features.to(device)
                gain = gain.to(device)
                vad = vad.to(device)

                pred_gain, pred_vad, states = model(features, states=states)
                states = [state.detach() for state in states]
                gain = gain[:,3:-1,:]
                vad = vad[:,3:-1,:]
                target_gain = torch.clamp(gain, min=0)
                target_gain = target_gain*(torch.tanh(8*target_gain)**2)

                e = pred_gain**gamma - target_gain**gamma
                gain_loss = torch.mean((1+5.*vad)*mask(gain)*(e**2))
                #vad_loss = torch.mean(torch.abs(2*vad-1)*(vad-pred_vad)**2)
                vad_loss = torch.mean(torch.abs(2*vad-1)*(-vad*torch.log(.01+pred_vad) - (1-vad)*torch.log(1.01-pred_vad)))
                loss = gain_loss + .001*vad_loss

                loss.backward()
                optimizer.step()
                if args.sparse:
                    model.sparsify()

                scheduler.step()

                running_gain_loss += gain_loss.detach().cpu().item()
                running_vad_loss += vad_loss.detach().cpu().item()
                running_loss += loss.detach().cpu().item()
                tepoch.set_postfix(loss=f"{running_loss/(i+1):8.5f}",
                                   gain_loss=f"{running_gain_loss/(i+1):8.5f}",
                                   vad_loss=f"{running_vad_loss/(i+1):8.5f}",
                                   )

        # save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'rnnoise{args.suffix}_{epoch}.pth')
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['loss'] = running_loss / len(dataloader)
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, checkpoint_path)
