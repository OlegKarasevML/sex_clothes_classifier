# Multiclass clothes classification and binary sex classification. 

## How to use
1) pull image from docker hub - docker pull olegkar/sex_clothes_classifier:project
2) launching on your machine -  docker run -d -p 5091:5091 olegkar/sex_clothes_classifier:project
3) go to a local server created by container -  http://localhost:5091/docs
4) choose classification_sex to classify an image, then Try it out, then you can choose your own image and lauch classifier.
5) choose classification_clothes to classify an image, then Try it out, then you can choose your own image and lauch classifier.

## Notebooks
1) Task_1 - solving train loop problem and retraining MobileNetV2 on datasets from Lamoda and Wildberries.
2) Task_2 - retraining sex classifier on celebrity dataset, quantization and 'mobilization' (MobileNetV2).
3) сlassification_clothes - retraining clothes classifier on datasets from Lamoda and Wildberries. 
4) distillation_classification -  attempt to retrain classifier MobileNetV3 on datasets from Lamoda and Wildberries, distilation and quantization. Unfortunately there was not enough GPU to retrain it to achive acceptible classification quality. 
