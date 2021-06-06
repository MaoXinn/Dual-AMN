# Boosting the Speed of Entity Alignment10×: Dual AttentionMatching Network with Normalized Hard Sample Mining (https://arxiv.org/pdf/2103.15452.pdf)

## Datasets

The datasets are from [GCN-Align](https://github.com/1049451037/GCN-Align), [JAPE](https://github.com/nju-websoft/JAPE), and [RSNs](https://github.com/nju-websoft/RSN).

* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* ref_ent_ids: entity links encoded by ids;
* triples_1: relation triples encoded by ids in source KG;
* triples_2: relation triples encoded by ids in target KG;

## Environment

* Python = 3.6
* Keras = 2.2.5
* Tensorflow = 1.14.0
* jupyter
* Scipy
* Numpy
* tqdm
* numba

## Acknowledgement

We refer to the codes of these repos: [keras-gat](https://github.com/danielegrattarola/keras-gat), [GCN-Align](https://github.com/1049451037/GCN-Align). 
Thanks for their great contributions!
