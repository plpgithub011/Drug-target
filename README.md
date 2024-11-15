# README for PREDICTING DRUG-TARGET BINDING AFFINITY WITH A BAGGING-BASED ENSEMBLE MODEL
## Abstract

The crucial problem of drug-target interaction (DTI) identification in the discovery of drug it  is addressed by the proposed study. The current experimental validations of drug-target binding profiles take a long time, hence computational tools are needed to speed up the process. The complex feature of binding strength is typically overlooked by existing computer approaches, which oversimplify by concentrating on binary classification. The article presents PREDICTING DRUG-TARGET BINDING AFFINITY WITH A BAGGING-BASED ENSEMBLE MODEL” a two-stage deep neural network ensemble model, as a solution for this problem. For the purpose of regression prediction, it uses a bagging-based ensemble learning technique and utilises sequence and structural information to create a fusion feature map. In particular, as compared to previous techniques, this model  shows a notable rise in Concordance Index (CI) on both the KIBA and Davis datasets. An route towards potentially finding novel Drug-Target Interactions (DTIs) is the in-silico screening, which is made possible by the approach.
Through public access to the codebase and dataset, drug discovery research can proceed with greater cooperation and investigation




## USAGE
### Required
- [tensorflow](https://www.tensorflow.org/)[1.15]
### Run

You can manually change the parameters in the go.sh file and execute ` ./go.sh`  for the first-stage training to extract efficient candidate protein and drug pair representation.Then run ` python catboost.py` to get the final prediction result.
 Before use  **catboost.py**, you need to generate **test_Y.txt** , **train_Y.txt** and **DenseFeature.h5** by **generate_h5File.ipynb** to avoid using gpu and speed up the training.

we can also directly run the file **catboost.py** in jupiternotebook to get output 