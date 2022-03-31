# README

Code for Two-stage Identifier: "**Parallel Instance Query Network for Named Entity Recognition**", accepted at ACL 2022. For details of the model and experiments, please see [our paper](https://arxiv.org/abs/2203.10545).

![](./assets/overview.jpg)

## Setup

### Requirements

```bash
conda create --name acl python=3.8
conda activate acl
pip install -r requirements.txt
```
### Datasets

Nested NER:

+ ACE04: https://catalog.ldc.upenn.edu/LDC2005T09
+ ACE05: https://catalog.ldc.upenn.edu/LDC2006T06
+ KBP17: https://catalog.ldc.upenn.edu/LDC2017D55
+ GENIA: http://www.geniaproject.org/genia-corpus
+ NNE: https://github.com/nickyringland/nested_named_entities

Flat NER:

+ OntoNotes: https://github.com/yhcc/OntoNotes-5.0-NER
+ CoNLL03: https://data.deepai.org/conll2003.zip
+ FewNERD: https://ningding97.github.io/fewnerd/
+ MSRA: https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/ner2mrc/download.md


Data format:
```json
{
    "tokens": ["Others", ",", "though", ",", "are", "novices", "."], 
    "entities": [{"type": "PER", "start": 0, "end": 1}, {"type": "PER", "start": 5, "end": 6}], "relations": [], "org_id": "CNN_IP_20030328.1600.07", 
    "ltokens": ["WOODRUFF", "We", "know", "that", "some", "of", "the", "American", "troops", "now", "fighting", "in", "Iraq", "are", "longtime", "veterans", "of", "warfare", ",", "probably", "not", "most", ",", "but", "some", ".", "Their", "military", "service", "goes", "back", "to", "the", "Vietnam", "era", "."], 
    "rtokens": ["So", "what", "is", "it", "like", "for", "them", "to", "face", "combat", "far", "from", "home", "?", "For", "an", "idea", ",", "here", "is", "CNN", "'s", "Candy", "Crowley", "with", "some", "war", "stories", "."]
}
```

The `ltokens` contains the tokens from the previous sentence. And The `rtokens` contains the tokens from the next sentence.

Due to the license, we cannot directly release our preprocessed datasets of ACE04, ACE05, KBP17, NNE and OntoNotes. We only release the preprocessed GENIA, FewNERD, MSRA and CoNLL03 datasets. Download them from [here](https://drive.google.com/drive/folders/1UttZVSL9iAqxsfPfMfSAl9FYc4DytfP5?usp=sharing). 

If you need other datasets, please contact me (`syl@zju.edu.cn`) by email. Note that you need to state your identity and prove that you have obtained the license.
## Example

### Train


```bash
python piqn.py train --config configs/nested.conf
```

Note: You should edit this [line](https://github.com/tricktreat/piqn/blob/e161cfa373ddd3e5162f71e260c7d6f3946eff33/config_reader.py#L21) in `config_reader.py` according to the actual number of GPUs. 

### Evaluation

You can download our [checkpoints on ACE04 and ACE05](https://drive.google.com/drive/folders/1rIgz4gBn_Na3KjFa21dEvHDY03fekwv3?usp=sharing), or train your own model and then evaluate the model. Because of the limited space of Google Cloud Drive, we share the other models in Baidu Cloud Drive, please download at [this link (code: js9z)](https://pan.baidu.com/s/1ULV2XEobLv8-LlwjkIy5ig).
```bash
python identifier.py eval --config configs/batch_eval.conf
```

If you use the checkpoints (ACE05 and ACE04) we provided, you will get the following results:


+ ACE05:

```
2022-03-30 12:56:52,447 [MainThread  ] [INFO ]  --- NER ---
2022-03-30 12:56:52,447 [MainThread  ] [INFO ]  
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]                  type    precision       recall     f1-score      support
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]                   PER        88.07        92.92        90.43         1724
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]                   LOC        63.93        73.58        68.42           53
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]                   WEA        86.27        88.00        87.13           50
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]                   GPE        87.22        87.65        87.44          405
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]                   ORG        85.74        81.64        83.64          523
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]                   VEH        83.87        77.23        80.41          101
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]                   FAC        75.54        77.21        76.36          136
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]  
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]                 micro        86.38        88.57        87.46         2992
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]                 macro        81.52        82.61        81.98         2992
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]  
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]  --- NER on Localization ---
2022-03-30 12:56:52,475 [MainThread  ] [INFO ]  
2022-03-30 12:56:52,496 [MainThread  ] [INFO ]                  type    precision       recall     f1-score      support
2022-03-30 12:56:52,496 [MainThread  ] [INFO ]                Entity        90.58        92.91        91.73         2991
2022-03-30 12:56:52,496 [MainThread  ] [INFO ]  
2022-03-30 12:56:52,496 [MainThread  ] [INFO ]                 micro        90.58        92.91        91.73         2991
2022-03-30 12:56:52,496 [MainThread  ] [INFO ]                 macro        90.58        92.91        91.73         2991
2022-03-30 12:56:52,496 [MainThread  ] [INFO ]  
2022-03-30 12:56:52,496 [MainThread  ] [INFO ]  --- NER on Classification ---
2022-03-30 12:56:52,496 [MainThread  ] [INFO ]  
2022-03-30 12:56:52,516 [MainThread  ] [INFO ]                  type    precision       recall     f1-score      support
2022-03-30 12:56:52,516 [MainThread  ] [INFO ]                   PER        97.09        92.92        94.96         1724
2022-03-30 12:56:52,516 [MainThread  ] [INFO ]                   LOC        76.47        73.58        75.00           53
2022-03-30 12:56:52,516 [MainThread  ] [INFO ]                   WEA        95.65        88.00        91.67           50
2022-03-30 12:56:52,516 [MainThread  ] [INFO ]                   GPE        92.93        87.65        90.22          405
2022-03-30 12:56:52,516 [MainThread  ] [INFO ]                   ORG        93.85        81.64        87.32          523
2022-03-30 12:56:52,516 [MainThread  ] [INFO ]                   VEH       100.00        77.23        87.15          101
2022-03-30 12:56:52,516 [MainThread  ] [INFO ]                   FAC        89.74        77.21        83.00          136
2022-03-30 12:56:52,516 [MainThread  ] [INFO ]  
2022-03-30 12:56:52,516 [MainThread  ] [INFO ]                 micro        95.36        88.57        91.84         2992
2022-03-30 12:56:52,517 [MainThread  ] [INFO ]                 macro        92.25        82.61        87.05         2992
```

+ ACE04

```
2021-11-15 22:06:50,896 [MainThread  ] [INFO ]  --- NER ---
2021-11-15 22:06:50,896 [MainThread  ] [INFO ]  
2021-11-15 22:06:50,932 [MainThread  ] [INFO ]                  type    precision       recall     f1-score      support
2021-11-15 22:06:50,932 [MainThread  ] [INFO ]                   VEH        88.89        94.12        91.43           17
2021-11-15 22:06:50,932 [MainThread  ] [INFO ]                   WEA        74.07        62.50        67.80           32
2021-11-15 22:06:50,932 [MainThread  ] [INFO ]                   GPE        89.11        87.62        88.36          719
2021-11-15 22:06:50,932 [MainThread  ] [INFO ]                   ORG        85.06        84.60        84.83          552
2021-11-15 22:06:50,932 [MainThread  ] [INFO ]                   FAC        83.15        66.07        73.63          112
2021-11-15 22:06:50,932 [MainThread  ] [INFO ]                   PER        91.09        92.12        91.60         1498
2021-11-15 22:06:50,933 [MainThread  ] [INFO ]                   LOC        72.90        74.29        73.58          105
2021-11-15 22:06:50,933 [MainThread  ] [INFO ]  
2021-11-15 22:06:50,933 [MainThread  ] [INFO ]                 micro        88.48        87.81        88.14         3035
2021-11-15 22:06:50,933 [MainThread  ] [INFO ]                 macro        83.47        80.19        81.61         3035
2021-11-15 22:06:50,933 [MainThread  ] [INFO ]  
2021-11-15 22:06:50,933 [MainThread  ] [INFO ]  --- NER on Localization ---
2021-11-15 22:06:50,933 [MainThread  ] [INFO ]  
2021-11-15 22:06:50,954 [MainThread  ] [INFO ]                  type    precision       recall     f1-score      support
2021-11-15 22:06:50,954 [MainThread  ] [INFO ]                Entity        92.56        91.89        92.23         3034
2021-11-15 22:06:50,954 [MainThread  ] [INFO ]  
2021-11-15 22:06:50,954 [MainThread  ] [INFO ]                 micro        92.56        91.89        92.23         3034
2021-11-15 22:06:50,954 [MainThread  ] [INFO ]                 macro        92.56        91.89        92.23         3034
2021-11-15 22:06:50,954 [MainThread  ] [INFO ]  
2021-11-15 22:06:50,954 [MainThread  ] [INFO ]  --- NER on Classification ---
2021-11-15 22:06:50,955 [MainThread  ] [INFO ]  
2021-11-15 22:06:50,976 [MainThread  ] [INFO ]                  type    precision       recall     f1-score      support
2021-11-15 22:06:50,976 [MainThread  ] [INFO ]                   VEH        94.12        94.12        94.12           17
2021-11-15 22:06:50,976 [MainThread  ] [INFO ]                   WEA        95.24        62.50        75.47           32
2021-11-15 22:06:50,976 [MainThread  ] [INFO ]                   GPE        95.60        87.62        91.44          719
2021-11-15 22:06:50,976 [MainThread  ] [INFO ]                   ORG        93.59        84.60        88.87          552
2021-11-15 22:06:50,976 [MainThread  ] [INFO ]                   FAC        93.67        66.07        77.49          112
2021-11-15 22:06:50,976 [MainThread  ] [INFO ]                   PER        97.11        92.12        94.55         1498
2021-11-15 22:06:50,976 [MainThread  ] [INFO ]                   LOC        84.78        74.29        79.19          105
2021-11-15 22:06:50,976 [MainThread  ] [INFO ]  
2021-11-15 22:06:50,976 [MainThread  ] [INFO ]                 micro        95.59        87.81        91.53         3035
2021-11-15 22:06:50,976 [MainThread  ] [INFO ]                 macro        93.44        80.19        85.87         3035
```


## Citation
If you have any questions related to the code or the paper, feel free to email `syl@zju.edu.cn`.

```bibtex
@inproceedings{shen-etal-2022-piqn,
    title = "Parallel Instance Query Network for Named Entity Recognition",
    author = "Shen, Yongliang  and
      Wang, Xiaobin  and
      Tan, Zeqi  and
      Xu, Guangwei  and
      Xie, Pengjun  and
      Huang, Fei and
      Lu, Weiming and
      Zhuang, Yueting",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2203.10545",
}
```