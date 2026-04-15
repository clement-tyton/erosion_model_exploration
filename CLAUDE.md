# first open : ef1410ef-59ab-4821-b044-11f8ef6a040a.json obtain from the metadat 
# Downloaded from the input of objectrain

# the json has obe key "balanced_tiles" and the object inside the key is a list of element (19752 elements)

# for each ele;ents we have :
# {
    # "imagery_file": "image_238ea38e-6e55-46be-8285-4592f1ba47aa_1536_1536.npz",
    # "mask_file": "mask_238ea38e-6e55-46be-8285-4592f1ba47aa_1536_1536.npz",
    # "count": 14,
    # "bands": ["MEP", "RED", "GREEN", "BLUE", "DSM", "DSM_NORMALIZED"]
    # }


  
# Got it its RED Green Blue and Normalised DSM -> available in modeltrain workflow inputs parameters

# the model is here 
#   "epoch_file_key": "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/1360284e-3545-40ae-96f1-031b14629f3d/model_epoch_80.pth",
# in order to have it on ntyton ai go to ObjectModeltrain/TestEpoch/Test_epoch_80/getmodelepoch  actvitity and tech it in the output !


# First Idea, open tile by tile maybe group them per bathc dataloader style without random 
# output predictions of the loaded model -> compare to the mask -> compute metrics (iou, f1 score, precision, recall) 
# for each tile its a metric by tile though !

# afer that order each tile by f1_score erosion its class 14 descending  -> allow us to understand  the false positive 
# rthen order by f1_scoreno_erosion decreasing too to understad tghe false positive !

# create some visualisation helpers in order to see the RGB image the DSM layer and also ->the maskpredit and the true mash  2x2 ?

# train mean and trqin std are on metadata_model_train.json file keys can be useful fror normalisation