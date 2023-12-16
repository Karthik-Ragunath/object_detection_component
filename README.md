# CS6301-Project
Object-Detection

--------------------------------

# Creating training and inference environments

## Requirements

Python Version - **Python 3.10**

### STEPS TO CREATE ENV

```
pip install -r requirements.txt
```

Train images must be placed in: `images` directory

Train labels must be placed in: `labels` directory

--------------------------------

# RUN TRAINING PROCESS

```
python train.py
```

--------------------------------

# RUN INFERENCE PROCESS

```
python inference.py
```

--------------------------------

# TRAINED MODEL

Trained model is located in:
`https://drive.google.com/drive/folders/1DAeYLMtmMXdzRHGvuuw8TLeVXJnkaYBb?usp=sharing`

Download it into `object_detection_component` directory

Since the size of this model is big, we are not adding it to zip file.

After downloading, the model will present in the filepath:
`object_detection_component/fasterrcnn_resnet50_fpn_custom.pth`

--------------------------------

# COLAB Notebook for training, inference and plotting the graphs

```
colab_notebook_train_inference_plotting.ipynb
```

--------------------------------
