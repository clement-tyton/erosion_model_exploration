#  TODO :
- impact of unbalanced data ? test 
- model train on curated dataset ? ie i get rid of the bad tiles , 
- Download the tst set Dampier, Dugong  Redtingle ?
- model train or modle fine tuning  ? which level of fine tuing ?
- dashboard on mlflow ?


- MODEL Train On 256x256..
- shall accept384 by 384 ?
- evaluation on train only, download the etset manually at some poinjt
- Recall maybe + accurate que le f1 score ,  I want to know how many positif I miss, not how many i do to much
- maybe an analysis finer than f1 score ?
    - recall what i miss in erosion  
    - precision what i see to much as hwerosion ?
very intuitivey the recall is 
lower than the precision
Recall₀ = TP₀ / (TP₀ + FP₁)   Recall₁ = TP₁ / (TP₁ + FP₀)   
FN₀ = FP₁ → erosion pixels falsely called no-erosion hurt recall₀
FN₁ = FP₀ → no-erosion pixels falsely called erosion hurt recall₁

# notses  in ordr to have the link on s3 of a model with its epochs
# go to object modeltrainworkflow / test epochs / select the epochs wished / getmodel epoch output json give the model on s3 .
# for the used tiles go to object train activities inside the object model train workflow and extract the pth

# build  a json lnklabel model on tytonai  -> s3 lnk ?

# dea one :
*cross the depth clculation with the model capacity in order to know if there is a link between performance of themodel anddepth  nuage de pont r**2
# dea 2 :
we need sometimes a bigger picture to see all the erosion, since the model is trained on256x256  based on a384x384 tiles max then -> it cant have this big picture !


Ok :
model_v1_jaswinder_epoch50.pth :
    -"balanced_tiles_path": "da2dabe0-eb9a-4238-8b8d-d1851df7b968.json",
    -   "epoch_file_key": "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/2424a7bc-28e6-4e94-a558-075e8c25264f/model_epoch_50.pth",
    

model_v2_no_erosion_td_epoch50.pth :
    - "balanced_tiles_path": "f8d5276b-0616-4f25-a580-801878ac0dc4.json", 
    -"epoch_file_key": "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/9f39f3c2-ce50-4c6b-a3ae-c355e29f77c8/model_epoch_50.pth",

model_v3_splt_test_epoch50.pth:
    "balanced_tiles_path": "17620a1a-8cb2-439e-96ef-b6131e34dda3.json" 
    -"epoch_file_key": "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/a1aacb5b-64b6-4069-939f-ca99f3879d89/model_epoch_50.pth",

model_v3_splt_test_epoch78.pth:
    "balanced_tiles_path": -ef1410ef-59ab-4821-b044-11f8ef6a040a.json"
    "epoch_file_key": "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/1360284e-3545-40ae-96f1-031b14629f3d/model_epoch_78.pth",

    
model_v3_splt_test_epoch80.pth:
    "balanced_tiles_path":ef1410ef-59ab-4821-b044-11f8ef6a040a.json"
    "epoch_file_key": "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/1360284e-3545-40ae-96f1-031b14629f3d/model_epoch_80.pth"
  

Model Comparison Tab 