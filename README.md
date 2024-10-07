# Multi-Modal Facebook Hateful Meme Classification

## Dataset - Facebook hateful meme data with offensive and non offensive labels.

source - https://www.kaggle.com/datasets/audreyhengruizhang/facebook-hateful-meme-captions 

## File 1 - Text Classification with BERT Sentence Transformers

### **Overview**

This project utilizes BERT Sentence Transformers for classifying meme texts based on offensive and non-offensive. The approach involves encoding the texts into sentence embeddings and then applying various machine learning classifiers to achieve accurate classification results.

### **Pretrained Sentence Transformer For Sentence Embeddings**

Importing the necessary libraries and load the pretrained BERT using SentenceTransformer.

`pythonfrom sentence_transformers import SentenceTransformer

*### Load the pretrained model*
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')`

### **Encode Text Data into Sentence Embeddings**

Next, encode the meme texts into sentence embedding using the loaded model.
`df['sent_bert'] = df['text'].apply(lambda x: sbert_model.encode(x))`

### **Visualization with T-SNE**

To visualize the separability of offensive and non-offensive text types. 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/47d4eabb-2253-40ae-94cb-d02846dd6d23/35b2f7e7-00ab-497b-9b33-b1c45b3616a3/image.png)

### **Classification Models**

Various machine learning classifiers are applied to classify the sentence embeddings:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Classifier**
- **Multi-layer Perceptron (MLP)**

### **Training and Evaluation**

Trained each model using the encoded sentence embedding and evaluated their performance on a test set. Below is an example of training a ML models:

| **Model** | **Accuracy** |
| --- | --- |
| Logistic Regression | ~79% |
| Decision Tree | ~82% |
| Random Forest | ~76% |
| Support Vector Classifier | ~81% |
| Multi-layer Perceptron | ~74% |

## **File 2 - Image Classification with VGG16 and ResNet**

### **Overview**

This project utilizes fine tuning VGG16 and ResNet architectures for classifying images, using Facebook Offensive and Non-Offensive meme images and their associated labels. 

### **Architectures of VGG16 and ResNet**

### **VGG16 Architecture**

VGG16 is a convolutional neural network (CNN) architecture known for its simplicity and depth. It consists of the following layers:

- **Input Layer**: Accepts images of size 224x224 pixels.
- **Convolutional Layers**:
    - Multiple layers of convolution with small filters (3x3) followed by ReLU activation.
    - Max pooling layers (2x2) after certain convolutional layers to reduce spatial dimensions.
- **Fully Connected Layers**:
    - A series of fully connected layers that flatten the output from the convolutional layers.
    - The final layer uses a sigmoid activation function for binary classification.
    

### **ResNet Architecture**

ResNet (Residual Network) is designed to overcome the vanishing gradient problem in deep networks by using skip connections. Its architecture includes:

- **Input Layer**: Accepts images of size 224x224 pixels.
- **Convolutional Layers**:
    - Initial convolution layer followed by batch normalization and ReLU activation.
    - Residual blocks that consist of two or three convolutional layers with skip connections.
- **Fully Connected Layers**:
    - Similar to VGG, the output is flattened and passed through fully connected layers with a final sigmoid activation for binary classification.

### **Training and Evaluation**

Trained both the model using the prepossessed image data and evaluated their performance on a test set

| **Model** | **Accuracy** |
| --- | --- |
| VGG16 | ~50% |
| ResNet | ~51% |

## **File 3 - Multimodal Classification with LSTM and CNN**

### **Overview**

This project employs a multimodal approach to classify memes as offensive and non offensive. The architecture integrates Long Short Term Memory (LSTM) networks for processing text and Convolutional Neural Networks (CNN) for analyzing images. This allows the model to leverage both modalities for classification.

### **Multimodal Model Architecture**

The model architecture consists of the following components:

- **Input Layer**:
    - Takes an embedding matrix as input, where each word is represented as a vector using the Word2Vec model.
- **Text Processing with LSTM**:
    - The LSTM model processes the input embedding matrix as a padded sequence of text corresponding to memes or inspirational quotes.
    - The architecture includes two layers of LSTM to capture temporal dependencies in the text data.
- **Image Processing with CNN**:
    - The CNN model processes images of memes or inspirational quotes.
    - A single convolutional layer is used to extract features from the images.
- **Concatenation Layer**:
    - Outputs from both the LSTM and CNN models are concatenated.
- **Fully Connected Layers**:
    - The concatenated output is fed into a couple of Dense layers with ReLU activation functions and Dropout layers set at 30% to prevent overfitting.
    - The final output layer uses a sigmoid activation function for binary classification.

### **Loss Function and Optimization**

- **Loss Function**: Binary Cross Entropy
- **Optimizer**: Adam optimizer

### **Training Configuration**

- **Batch Size**: 100
- **Epochs**: 20
- **Early Stopping**: Patience set to 5 epochs without improvement.

### **Results Summary**

The shallow multimodal classifier demonstrated outstanding performance due to the additive nature of combining text and image outputs. The table below summarizes the performance metrics:

| **Metric** | **Value** |
| --- | --- |
| Training Accuracy | ~71% |
| Validation Accuracy | ~55% |
