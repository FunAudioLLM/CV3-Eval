import sys, os
import torch
from tqdm import tqdm
import multiprocessing
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration 
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel
import whisper
punctuation_all = punctuation + string.punctuation

wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3] # zh or en

device = "cuda" if torch.cuda.is_available() else 'cpu'

def load_en_model():
    model_id = "whisper-large-v3/large-v3.pt"
    model = whisper.load_model(model_id).to(device)
    model.eval()
    return model

def load_zh_model():
    model = AutoModel(model="paraformer-zh")
    return model

def process_one(hypo, truth):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')

    if lang[-2:] in ["zh", "ja", "ko"]:
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo]) # 中文hypo自带空格
    else:
    # elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    # else:
        # raise NotImplementedError

    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    return (raw_truth, raw_hypo, wer, subs, dele, inse)


def run_asr(wav_res_text_path, res_path):
    if lang[-2:] in ["zh", "hard_zh"]:
        model = load_zh_model()
    else:
        model = load_en_model()

    params = []
    for line in open(wav_res_text_path).readlines():
        line = line.strip()
        if len(line.split('|')) == 2:
            wav_res_path, text_ref = line.split('|')
        elif len(line.split('|')) == 3:
            wav_res_path, wav_ref_path, text_ref = line.split('|')
        elif len(line.split('|')) == 4: # for edit
            wav_res_path, _, text_ref, wav_ref_path = line.split('|')
        else:
            raise NotImplementedError

        if not os.path.exists(wav_res_path):
            continue
        params.append((wav_res_path, text_ref))
    fout = open(res_path, "w")

    n_higher_than_50 = 0
    wers_below_50 = []
    for wav_res_path, text_ref in tqdm(params):
        try:
            if lang[-2:] in ["zh", "hard_zh"]:
                res = model.generate(input=wav_res_path,
                        batch_size_s=300)
                transcription = res[0]["text"]
            else:
                result = model.transcribe(wav_res_path, language=lang[-2:])
                transcription = result["text"].strip()
        except Exception as e:
            print(e)
            continue
        if 'zh' in lang:
            transcription = zhconv.convert(transcription, 'zh-cn')
            
        raw_truth, raw_hypo, wer, subs, dele, inse = process_one(transcription, text_ref)
        fout.write(f"{wav_res_path}\t{wer}\t{raw_truth}\t{raw_hypo}\t{inse}\t{dele}\t{subs}\n")
        fout.flush()

run_asr(wav_res_text_path, res_path)

