from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import json
from pycocotools.coco import COCO
import nltk

# Load ground truth
annFile = '/Users/umangshikarvar/Desktop/Project/coco/annotations/captions_val2017.json'
coco = COCO(annFile)
img_ids = coco.getImgIds()
ref_dict = {img_id: [ann['caption'] for ann in coco.imgToAnns[img_id]] for img_id in img_ids}

# Load your predictions
with open("/Users/umangshikarvar/Desktop/Project/predictions.json") as f:
    preds = json.load(f)

references = []
hypotheses = []

for p in preds:
    img_id = p["image_id"]
    hyp = nltk.word_tokenize(p["caption"].lower())
    refs = [nltk.word_tokenize(cap.lower()) for cap in ref_dict[img_id]]
    hypotheses.append(hyp)
    references.append(refs)

smooth = SmoothingFunction().method4

print("BLEU-1:", corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smooth))
print("BLEU-2:", corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
print("BLEU-3:", corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth))
print("BLEU-4:", corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))