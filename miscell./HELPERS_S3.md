source .env && aws s3 cp "s3://${S3_FILE_BUCKET}/ef1410ef-59ab-4821-b044-11f8ef6a040a.json" . \
  --endpoint-url "https://${AWS_S3_ENDPOINT}" \
  --region "$AWS_REGION" \
  --profile session_token_profile


source .env && aws s3 cp "s3://${S3_FILE_BUCKET}/image_cca2e34e-a60c-49a9-9e21-a2524b083780_1536_1152.npz" "test.npz"

source .env && aws s3 cp "s3://${S3_FILE_BUCKET}/f9479ddf-33ee-11f1-b626-1706c5c76e3d" data/training_data/ \
  --endpoint-url "https://${AWS_S3_ENDPOINT}" \
  --region "$AWS_REGION" \
  --profile session_token_profile \
  --recursive


source .env && aws s3 ls "s3://${S3_FILE_BUCKET}/5d75e2df-33ef-11f1-b626-f313592c7d6c/" \
  --endpoint-url "https://${AWS_S3_ENDPOINT}" \
  --region "$AWS_REGION" \
  --profile session_token_profile \
  --recursive

# the model is here 
"epoch_file_key": "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/1360284e-3545-40ae-96f1-031b14629f3d/model_epoch_80.pth",
in order to have it on ntyton ai go to 
ObjectModeltrain/TestEpoch/
    Test_epoch_80/
        getmodel_epoch  actvitity and take it in the output !

mkdir -p models && source .env && aws s3 cp "s3://${S3_FILE_BUCKET}/1360284e-3545-40ae-96f1-031b14629f3d/model_epoch_80.pth" models/ \
  --endpoint-url "https://${AWS_S3_ENDPOINT}" \
  --region "$AWS_REGION" \
  --profile session_token_profile


# download single tile (imagery + mask)

source .env && aws s3 cp "s3://${S3_FILE_BUCKET}/mask_ec3bdbe9-0185-4997-970a-6d5b4fe3cd93_1920_384.npz" data/training_data/ \
  --endpoint-url "https://${AWS_S3_ENDPOINT}" \
  --region "$AWS_REGION" \
  --profile session_token_profile


# MMV1
   "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/2424a7bc-28e6-4e94-a558-075e8c25264f/model_epoch_50.pth",

# MMV2
  "epoch_file_key": "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/9f39f3c2-ce50-4c6b-a3ae-c355e29f77c8/model_epoch_50.pth",

# MMV3 split test Element_Geospatial_Erosion_MMv3_split_test
  "epoch_file_key": "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/a1aacb5b-64b6-4069-939f-ca99f3879d89/model_epoch_50.pth",


# MMV3 split test Element_Geospatial_Erosion_MMv3_split_test epoch80 
"epoch_file_key": "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/1360284e-3545-40ae-96f1-031b14629f3d/model_epoch_80.pth",
 
# MMV3 split test Element_Geospatial_Erosion_MMv3_split_test epoch 78
 "epoch_file_key": "s3://9b38a3ce-6fd6-11ef-8285-472724185bb4/1360284e-3545-40ae-96f1-031b14629f3d/model_epoch_78.pth",


# ok you will write for me multiple bash command in order to bring this  5 models in the models foldeer ->
# nom des models doit inclure info 
- v1 = (Jaswinder)
- v2 = v1 + no_erosion_TDv # DL the TD ausssi du modele ? YES
- v3 = v2 + no_erosion_training_data + split test -> 50, 78, and 80 epochs 


# comparaion de smodeles tableau erecapitulatif dans un autre onglet avce les 5 implique de relancer levaluation pour ces 5 modeles ok ? les memes statisitques que dhqb ? je veux voir clqremet leque;l est le meilelur if any
(noter qulque pqrt que je fqis linference sur itile 384  dans lapp )
