# Emotional Face Recognition through Machine Learning

**02476 Machine Learning Operations**

- Anna Ekner (s193396)
- Danyu Shen (s204165)
- Johan Verrecchia (s204127)
- Hans Christian Hansen (s103629)
- Jan Dufek (s240485)

## Project description

In the project, the aim is to explore the principles of 02476 Machine Learning Operations (MLOps) to tackle the challenge of recognizing emotions. Utilizing the proposed dataset [Facial Emotion Recognition Image Dataset](https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset?resource=download) from Kaggle as the data foundation for the project, the objective is to implement various machine learning models and techniques, to analyze and interpret emotional features effectively. The dataset consists of 15453 images of human faces categorized into six different emotions: happy, angry, sad, neutral, surprise, and ahegao.

The focus will be on integrating machine learning models into a production setting efficiently. This includes aspects such as code organization, deployment strategies through git repositories and containerized applications, reproducibility, and version control for collaborative development. Furthermore, the project will employ system monitoring and continuous integration to assess and enhance model performance. All of the mentioned will be applied in the quest for scalable and maintainable machine learning operations.

## Frameworks

The project will be built using Pytorch in combination with other third-party frameworks.
For the third-party framework, both PyTorch Image Models (timm) and Albumentations could be useful libraries in that they both excel in image-processing tasks. [Albumentations](https://github.com/albumentations-team/albumentations) could be used to balance the dataset, since there is currently some class imbalance between the six classes. [Timm](https://github.com/huggingface/pytorch-image-models) provides a large variety of models to use which also includes pretrained weights - also it's flexibility lies in the numbers of architecture it offers.

Both frameworks can be used in the same project, where timm is used for the model, and Albumentations is used for it's augmentations [stackoverflow](https://stackoverflow.com/questions/71476099/how-to-add-data-augmentation-with-albumentation-to-image-classification-framewor).

These frameworks will be incorporated into our workflow using Docker containers, ensuring a consistent and reproducible environment that facilitates easy sharing among team members.

## Expected models

The project will explore a range of machine learning algorithms, emphasizing convolutional neural networks (CNNs). This will include optimizing the layers and parameters of the CNN to enhance its generalization of patterns in human emotions. Techniques such as fine-tuning and regularization will be employed to improve the model's learning capabilities. However, the main focus of this project is the machine learning operations, not the machine learning model itself.
