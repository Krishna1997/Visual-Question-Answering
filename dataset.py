import os
import operator
import numpy as np

from collections import defaultdict
from vqa import VQA
def dd():
    return defaultdict(lambda: len(q2i))


def pre_process_dataset(image_dir, qjson, ajson, img_prefix):
    print('Preprocessing datatset. \n')
    vqa = VQA(ajson, qjson)

    img_names = [f for f in os.listdir(image_dir) if '.jpg' in f]
    img_names = img_names[:30000]
    print("length: ",len(img_names))
    img_ids = []
    for fname in img_names:
        img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
        img_ids.append(int(img_id))
    print("Done collecting image ids")
    ques_ids = vqa.getQuesIds(img_ids)

    q2i = defaultdict(lambda: len(q2i))
    
    pad = q2i["<pad>"]
    start = q2i["<sos>"]
    end = q2i["<eos>"]
    UNK = q2i["<unk>"]

    a2i_count = {}
    for ques_id in ques_ids:
        qa = vqa.loadQA(ques_id)[0]
        qqa = vqa.loadQQA(ques_id)[0]

        ques = qqa['question'][:-1]
        [q2i[x] for x in ques.lower().strip().split(" ")]
	
        answers = qa['answers']
        for ans in answers:
            if not ans['answer_confidence'] == 'yes':
                continue
            ans = ans['answer'].lower()
            if ans not in a2i_count:
                a2i_count[ans] = 1
            else:
                a2i_count[ans] = a2i_count[ans] + 1
    print("Done collecting Q/A")
    a_sort = sorted(a2i_count.items(), key=operator.itemgetter(1), reverse=True)

    i2a = {}
    count = 0
    a2i = defaultdict(lambda: len(a2i))
    for word, _ in a_sort:
        a2i[word]
        i2a[a2i[word]] = word
        count = count + 1
        if count == 1000:
            break
    print("Done collecting words")
    return q2i, a2i, i2a, a2i_count

if __name__ == '__main__':
    image_dir = "./data/train2014"
    img_prefix = "COCO_train2014_"
    qjson = "./data/v2_OpenEnded_mscoco_train2014_questions.json"
    ajson = "./data/v2_mscoco_train2014_annotations.json" 

    q2i, a2i, i2a, a2i_count = pre_process_dataset(image_dir, qjson, ajson, img_prefix)
    np.save('./data/q2i.npy', dict(q2i))
    np.save('./data/a2i.npy', dict(a2i))
    np.save('./data/i2a.npy', dict(i2a))
    np.save('./data/a2i_count.npy', dict(a2i_count))


