Requirements:
indic-nlp-library=0.92
sacrebleu=2.4.2
unbabel-comet=2.2.2

Commands:
1. Command to get TER Score:
python3 ter.py --ref reference.txt --pred predictions.txt

2. Command to get BLEU and CHRF Scores:
python3 bleu_chrf.py --ref reference.txt --pred predictions.txt

3. Command to get COMET scores (Using 'Unbabel/wmt22-comet-da' model):
CUDA_VISIBLE_DEVICES=0 python3 comet.py --src source.txt --pred predictions.txt --ref reference.txt
