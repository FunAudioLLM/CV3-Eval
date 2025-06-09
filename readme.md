# CV3-Eval
To address the challenges of diversity and generalizability from unrestrained real-world speech
synthesis scenarios, Cosyvoice 3 releases CV3-Eval benchmark for zero-shot speech synthesis in the
wild, which is built on authentic in-the-wild reference speech from Common Voice, FLUERS,
EmoBox, and Web-crawled real-world audio data, and spans a broad range of languages and
dialects, domains and environments, emotions and styles.

## Dataset
CV3-Eval includes subsets for both objective and subjective evaluation. The objective evaluation subset is further split into three subsets, including multilingual voice cloning, cross-lingual voice cloning, and emotion cloning. Three subjective subsets are prepared for expressive voice cloning, expressive voice continuation, and Chinese accent voice cloning. You can find more details in the [Cosyvoice 3](https://arxiv.org/pdf/2505.17589) paper.


## Metrics
We focus on three key aspects: content consistency, speaker similarity, and audio quality.    
* For content consistency, we measure the Character Error Rate (CER) or Word Error Rate (WER) of the ASR transcription against the given text, using Whisper-large V3 for English ASR and Paraformer for Chinese ASR. 
* To assess speaker similarity, we extract speaker embeddings from the generated speech using the ERes2Net speaker verification model and calculate the cosine similarity with the embedding of the reference speech. 
* For audio quality, we score the generated speech using the DNSMOS network, the scores of which show high correlations with human auditory perception.


## Requirements
```bash
conda create -n cv3-eval python=3.10
conda activate cv3-eval

# Install pytorch with your CUDA version, e.g.
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```


## Evaluation Code
We release the evaluation codes for all metrics:
```bash
# remember to modify <decode_dir> to the path of your own generated audio

# WER, speaker similarity, and DNSMOS
bash run_infer_cv3_eval.sh

# emotion score
bash run_infer_cv3_eval_emo.sh
```

## Citation
If this benchmark is useful for you, please cite as:
```
@article{du2025cosyvoice,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal={arXiv preprint arXiv:2505.17589},
  year={2025}
}
```