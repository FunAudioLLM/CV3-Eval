import sys
import librosa
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from sklearn.metrics import classification_report

wav_scp = sys.argv[1]
output_file = sys.argv[2]

labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unk']

inference_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="utils/emo_eval/model/emotion2vec_plus_large/")

hyp_high_list = []
ref_high_list = []

hyp_low_list = []
ref_low_list = []

with open(wav_scp, 'r') as f, open(output_file, 'w') as wf:
    for line in f:
        sc = '\t' if '\t' in line else ' '
        wid, path = line.strip().split(sc)
        y, sr = librosa.load(path, sr = 16000)

        rec_result = inference_pipeline(y, granularity="utterance", extract_embedding=False)
        scores = rec_result[0]['scores']
        hyp_emo = labels[scores.index(max(scores))]
        ref_emo, level = wid.split('_')[0], wid.split('_')[1]

        if level == 'high':
            hyp_high_list.append(hyp_emo)
            ref_high_list.append(ref_emo)
        else:
            hyp_low_list.append(hyp_emo)
            ref_low_list.append(ref_emo)

        wf.write(f"{wid}\t{ref_emo}\t{hyp_emo}\n")

print("====Score for high====")
print(classification_report(ref_high_list, hyp_high_list, digits=3))

print("====Score for low====")
print(classification_report(ref_low_list, hyp_low_list, digits=3))

print("====Score for all====")
print(classification_report(ref_high_list + ref_low_list, hyp_high_list + hyp_low_list, digits=3))
