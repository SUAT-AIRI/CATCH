# CATCH: A Controllable Theme Detection Framework with Contextualized Clustering and Hierarchical Generation #

### Brief Description ###
CATCH adopts a theme generation process to perform theme detection, which is made up of three main modules involving 1. Context-aware theme representation via topic segmentation (TopSeg); 2. Preference-enhanced topic clustering (PeC); 3. Hierarchical theme label generation (HieGen). 

To perform CATCH, user should execute the python files from the folders sequencially following the role: 1. TopSeg, 2. PeC, 3.HieGen.




## TopSeg ##
This stage involves two training process, and will provide a single json file that indicates the dialogues segmentation result (slice_file).

### step 1. Raw Data Preprocessing: ###

```
python data_preprocess.py  --dataset --save_name
```

### step 2. Conversation-Level Adaption ###

2.1 Model Training
```
torchrun DDP_train.py --dataset dstc --save_model_name dstc --epoch 4
```

### step 3. Utterance-Level Adaption ###

3.1 Preference-data preprocessing 
```
python preference_data_preprocess.py --dataset --preference_data --output_dir
```

3.2 Model Training
```
torchrun DDP_train.py --dataset dstc_preference --save_model_name dstc_preference --epoch 6
```

### step 4. Dialogue Segementation ###

```
python boundary_inference.py --model --dataset --save_name
```




## PeC ##
This stage first trains the preference-reward-model (PRM) and use PRM to inference the preference scalar between each topic pairs in matrix tensor form. Then, this stage performs preference-enhanced clustering to obtain the topic clusters in json form.

### step 1. Preference-scalar Acquiring ####

1.1 PRM training 
```
python train_PRM.py  --model_path  --save_path --org_dataset --slice_file --pref_file
```
where slice_file is provided by TopSeg which indicates the dialogue segmentation, and pref_file is the original preference annotation.

1.2 Preference-scalar inference
```
python preference_generation.py --org_dataset --slice_data --model
```
Output the score_matrix

### step 2. Preference-enhanced Topic Clustering ####
```
clustering.py --org_dataset_file --slice_file --score_matrix
```
where score_matrix refers to the preference scalar of each two topics.
Output the topic clusters (clusters_with_preferences)





## HieGen ##
This stage takes topic clusters as input, and generate the theme for each cluster, thus assigning theme for each targer utterance.

```
theme_labeling.py --org_dataset_file --clusters_with_preferences --result_file
```
where "clusters_with_preferences" is provided by the PeC