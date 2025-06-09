# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download pretrained models from modelscope (https://www.modelscope.cn/models)
based on the given model id, and extract embeddings from input audio.
Please pre-install "modelscope".
Usage:
    1. extract the embedding from the wav file.
        `python infer_sv.py --model_id $model_id --wavs $wav_path `
    2. extract embeddings from two wav files and compute the similarity score.
        `python infer_sv.py --model_id $model_id --wavs $wav_path1 $wav_path2 `
    3. extract embeddings from the wav list.
        `python infer_sv.py --model_id $model_id --wavs $wav_list `
"""

import os
import sys
import re
import pathlib

import kaldiio
import numpy as np
import argparse
import torch
import torchaudio
from tqdm import tqdm

try:
    from speakerlab.process.processor import FBank
except ImportError:
    sys.path.append('%s/../..'%os.path.dirname(__file__))
    from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import
import math

parser = argparse.ArgumentParser(description='Extract speaker embeddings.')
parser.add_argument('--model_id', default='', type=str, help='Model id in modelscope')
parser.add_argument('--ref_wavs', default='', type=str, help='reference wavs')
parser.add_argument('--prompt_wavs', default='', type=str, help='reference wavs')
parser.add_argument('--hyp_wavs', default='', type=str, help='reference wavs')
parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model dir')
parser.add_argument('--log_file', default='', type=str, help='file to save log')
parser.add_argument('--devices', default="0", type=str)
parser.add_argument('--task_id', default=0, type=int)

CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ResNet_aug.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_Base_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Large_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 64,
    },
}

supports = {
    'damo/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2',
        'model': CAMPPLUS_VOX,
        'model_pt': 'campplus_voxceleb.bin',
    },
    'damo/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0',
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    'damo/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2',
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    'damo/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.4',
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    'damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.1',
        'model': ERes2Net_Base_3D_Speaker,
        'model_pt': 'eres2net_base_model.ckpt',
    },
    'damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0',
        'model': ERes2Net_Large_3D_Speaker,
        'model_pt': 'eres2net_large_model.ckpt',
    },
}

def main():
    args = parser.parse_args()
    assert args.model_id in supports, "Model id not currently supported."
    conf = supports[args.model_id]

    pretrained_model = os.path.join(args.local_model_dir, args.model_id.split('/')[1], conf['model_pt'])
    #pretrained_state = torch.load(args.model_id, map_location='cpu')
    print(pretrained_model)
    pretrained_state = torch.load(pretrained_model, map_location='cpu')

    # load model
    model = conf['model']
    # embedding_model = dynamic_import(model['obj'])(**model['args'])
    from speakerlab.models.eres2net.ERes2Net import ERes2Net as NET
    embedding_model = NET(**model['args'])
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    devices = args.devices.strip().split(',')
    task_id = int(args.task_id)
    device = f"cuda:{devices[task_id % len(devices)]}"
    embedding_model = embedding_model.to(device)

    def load_wav(wav_file, obj_fs=16000):
        if ".ark:" not in wav_file:
            wav, fs = torchaudio.load(wav_file)
            if wav.shape[0] > 1:
                wav = wav[0, :].unsqueeze(0)
        else:
            if ' ' in wav_file:
                fs = int(wav_file.split(' ')[1])
                wav_file = wav_file.split(' ')[0]
            else:
                fs = None
            retval = kaldiio.load_mat(wav_file)
            if isinstance(retval, tuple):
                if isinstance(retval[0], int):
                    fs, wav = retval
                else:
                    wav, fs = retval
            else:
                wav, fs = retval, fs if fs is not None else 16000

            if wav.dtype == np.int16:
                wav = wav / (2**16 - 1)
            elif wav.dtype == np.int32:
                wav = wav / (2**32 - 1)

            wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

        if fs != obj_fs:
            # print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
            wav, fs = torchaudio.sox_effects.apply_effects_tensor(
                wav, fs, effects=[['rate', str(obj_fs)]]
            )

        return wav

    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    def compute_embedding(wav_file):
        # load wav
        wav = load_wav(wav_file)
        # compute feat
        feat = feature_extractor(wav).unsqueeze(0).to(device)
        # compute embedding
        with torch.no_grad():
            embedding = embedding_model(feat).detach().cpu().numpy()

        return embedding

    # extract embeddings
    print(f'[INFO]: Calculate similarities...')
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    out_file = open(args.log_file, "wt")
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    ref_wavs = {}
    if args.ref_wavs is not None and len(args.ref_wavs) > 0 and os.path.exists(args.ref_wavs):
        for line in open(args.ref_wavs, "rt").readlines():
            uttid, wav_path = line.strip().split(maxsplit=1)
            ref_wavs[uttid] = wav_path
    prompt_wavs = {}
    for line in open(args.prompt_wavs, "rt").readlines():
        uttid, wav_path = line.strip().split(maxsplit=1)
        prompt_wavs[uttid] = wav_path
    hyp_wavs = []
    for line in open(args.hyp_wavs, "rt").readlines():
        uttid, wav_path = line.strip().split(maxsplit=1)
        hyp_wavs.append((uttid, wav_path))
    ref_scores, hyp_scores = [], []
    for uttid, hyp_wav_path in hyp_wavs:
        if uttid not in prompt_wavs:
            continue
        try:
            prompt_emb = compute_embedding(prompt_wavs[uttid])

            ref_score = 0
            if uttid in ref_wavs:
                ref_wav_path = ref_wavs[uttid]
                ref_emb = compute_embedding(ref_wav_path)
                ref_score = similarity(torch.from_numpy(prompt_emb), torch.from_numpy(ref_emb)).item()

            hyp_emb = compute_embedding(hyp_wav_path)
            hyp_score = similarity(torch.from_numpy(prompt_emb), torch.from_numpy(hyp_emb)).item()

            if math.isnan(ref_score) or math.isnan(hyp_score) or math.isinf(ref_score) or math.isinf(hyp_score):
                print(f"Warning: {uttid}: ref_score: {ref_score}, hyp_score: {hyp_score}")
            else:
                ref_scores.append(ref_score)
                hyp_scores.append(hyp_score)
                out_file.writelines(f"{uttid} {ref_score*100.0:.2f} {hyp_score*100.0:.2f}\n")
                out_file.flush()
        except:
            continue

    avg_ref_score = np.array(ref_scores).mean()
    avg_hyp_score = np.array(hyp_scores).mean()
    out_file.writelines(f"avg {avg_ref_score*100.0:.2f} {avg_hyp_score*100.0:.2f}\n")
    out_file.close()


if __name__ == '__main__':
    main()
