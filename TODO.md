# Work Session : Thursday  and friday, Goal first 2 simple experiment(Unet segformer and batch size,e nough fro mlflow) + nice mlflow dashboard
1) Build train and test dataset based on the JSON  the tran comes from my last training
2) Define some training metrics similar to tyton ai, and a common databalcne process  ? easy we have the file already
3) To test : segformer unet, partial weightfinetuning, tile size is impossible but could be 
let as an openning ! 2 slides presentation of the whole project
3) Plug mlflow S3 this time experiment Erosion ..
4) Dashb
5) I train a fne tuned model, add it on the loop ! to test  !  and go nin the app ! 
6) Gather information in the objec train workflow
- Question : OFrom where the base erosion model comes from whe we fine tune ? 
- model train 400 in excalidraw
7) for my own evaluation, compute the f1 score likethis, take your test set 384x384 tiles Damper dungong, and predict for each tie the model pred, and morehonnest from the mpodel perspective , more faire for the model 
8) interesting TO mesure  -> what is the f1 socre deprecation cause by the sze we do nference on tilesize which is not the same thant the one of the model, ?? Another Jira task -> segmetation classify activities
9) je vais classifier sur du 256  Mais vaudrait le coup de tester laleventuelle degradation enn passant de 512 (done in tyton ai) a (384 ) parailleurs parce qu ja les tuiles 384 seule;ment et je vais pas les cropper.. ca se test sur tyon ai !
pas urgent because lecart est pas sigrand, mais ce serait marrant de verifer limpact sur nos metriques
du coup on pourrait juste diviser  + et go 
SUper interesting the finetuning is done over 6000 tiles only 5 epochs thats not a lot ! master tyton ai configuration! mais on a que ses resultats sur le ties utilsees, common tiles feature ?
10) quelques exemple de f1 no erosion  bad to undrsand where a re the false NEGATIVE !!
 Marrant ces effets de bords dailleur son a vraiment certaines tiles tres pettes qi sont pqdees, ya til des controles faits a lentrainement ?


 # Good example : image_16413f59-4e3e-4018-8363-73a5e9cce2ae_1920_1920.npz

# On the use of normalsation param  (dataleak if the est set is used to compute the mean and var but shoul not be the case)

train_mean and train_std computed by data_balance from the training tiles only are stored in final_config and reused verbatim at inference. The test/inference data never participates in their computation. Exactly the correct "fit on train, apply everywhere" pattern.

One subtlety worth noting: max_pixel_value=1.0 means the raw pixel values are not divided by 255 first — the mean and std are in the 0–255 scale (as you can see from your real example: train_mean=[150.7, 123.7, 92.5, -9.7]). This needs to be consistent between training and inference, which is what the preprocess_before_augmentation flag governs and why a mismatch there would silently break things.

#  TODO :
- impact of unbalanced data ? test 
- model train on curated dataset ? ie i get rid of the bad tiles , 
- Download the tst set Dampier, Dugong  Redtingle ?
- model train or modle fine tuning  ? which level of fine tuing ?
- dashboard on mlflow ?

Your colleagues described it perfectly. The code confirms exactly that:


┌──────────────────────────────────┐
│         512 px tile              │  ← what you asked for
│  ┌────────────────────────────┐  │
│64│                            │64│  ← 64px border pixels borrowed from neighbours
│  │      512 px core           │  │
│  │                            │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
         640 px → model input
Before forward pass — each 512 px tile is extended by 64 px on each side using neighboring pixels → model sees 640×640
After prediction — the 64 px border of the prediction is discarded (line 326-328):

predicted_tile_batches[:, 64:-64, 64:-64, :]
→ back to 512×512
So my earlier table was wrong. The correct picture:

classify_tile_size	Model input	Output kept	vs training (256)
256	256 + 2×64 = 384	256	1.5×
512	512 + 2×64 = 640	512	2.5×
The overlap is not a "seam blending" — it's purely to give the model enough context at each tile boundary so edge predictions are not degraded. Only the central prediction is trusted.

# Example of segmentation classsify inut values :
{
  "bbox_grid": "b4ec8f2b-12d6-4cee-93cb-f0bc47a6123d/tile_grid.fgb",
  "bbox_index": 0,
  "checkpoint": "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/a1aacb5b-64b6-4069-939f-ca99f3879d89/model_epoch_50.pth",
  "class_list": [
    1,
    14
  ],
  "classify_tile_overlap_size": 64,
  "classify_tile_size": 512,
  "clip_file": "3c30178e-e2c4-4048-b1b2-db0f407a4636/Redtingle.fgb",
  "config": {
    "activation": "softmax2d",
    "bands": [
      "RED",
      "GREEN",
      "BLUE",
      "DSM_NORMALIZED"
    ],
    "encoder_depth": 5,
    "encoder_type": "timm-res2net101_26w_4s",
    "encoder_weights": "imagenet",
    "model_type": "Unet",
    "preprocess_before_augmentation": true,
    "train_mean": [
      150.73301134918557,
      123.75755228360018,
      92.57823716578613,
      -9.734063808604613
    ],
    "train_std": [
      39.721974708734216,
      34.06117915518031,
      30.092062243775406,
      4.684211737168346
    ]
  },
  "model_class_list": [
    1,
    14
  ],
  "padding": 128,
  "rasters": [
    {
      "bands": [
        "RED",
        "GREEN",
        "BLUE",
        "ALPHA"
      ],
      "raster_file": "108a84b0-6f03-40cb-bc73-9534a143b55e/RED_GREEN_BLUE_ALPHA_webmap.tif"
    },
    {
      "bands": [
        "DSM"
      ],
      "raster_file": "8751d69d-4378-4517-ad51-8ff2e032e438/DSM_webmap.tif"
    },
    {
      "bands": [
        "DSM_NORMALIZED"
      ],
      "raster_file": "5f4b233b-e077-410e-ae31-542279c6497c/DSM_NORMALIZED_webmap.tif"
    },
    {
      "bands": [
        "MEP"
      ],
      "raster_file": "546fe197-b432-430d-97b5-e5dcc90ade0d/MEP_webmap.tif"
    }
  ],
  "sieve_connectivity": 4,
  "sieve_size": 8
}

- MODEL Train On 256x256..
- shall accept 384 by 384 ?
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