import sys, os
from tqdm import tqdm

metalst = sys.argv[1]
wav_dir = sys.argv[2]
wav_res_ref_text = sys.argv[3]
print(metalst,wav_dir,wav_res_ref_text)
f = open(metalst)
lines = f.readlines()
f.close()

f_w = open(wav_res_ref_text, 'w')
for line in tqdm(lines):
    utt, infer_text = line.strip().split(maxsplit=1)
    if not os.path.exists(os.path.join(wav_dir, utt + '.wav')):
        continue
    out_line = '|'.join([os.path.join(wav_dir, utt + '.wav'), infer_text])
    f_w.write(out_line + '\n')
f_w.close()
