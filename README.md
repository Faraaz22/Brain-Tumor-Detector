
## **Project Overview**

This project is a deep learning-based image classification application that classifies images into two categories: "tumorous" and "non-tumorous." The model uses a pre-trained VGG19 architecture, fine-tuned with custom layers to perform binary classification. It demonstrates the use of deep learning (specifically transfer learning with VGG19) for image classification in a web application. It includes data preprocessing, model training, fine-tuning, and deploying the model using Flask for user interaction.


---
### **Steps to Run the Application**

1. Run ***Brainmodel.ipynb*** in jupyter notebook with the data set to create the Model Weights

2. Install the dependencies by running:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```
   flask run
   ```

4. Visit the application in a browser (typically at `http://127.0.0.1:5000/`).
5. Upload an image and receive the classification result.

---

## **Project Directory Structure**

The project is organized into the following structure:

```
image_classification/
│
├── app.py                    
├── static/                   
│   └── style.css             
│
├── templates/                
│   └── index.html            
│
├── uploads/                  
│   └── (user uploaded images) 
│
├── model_weights/            
    └── vgg_unfrozen.weights.h5 
│   └── vgg19_model_01.weights.h5  
│   └── vgg19_model_02.weights.h5 
│
├── model.keras               
│
└── requirements.txt         
```

## **Machine Learning Model Explanation**

This project uses **VGG19**, a deep convolutional neural network architecture, for the classification of images into two categories: **tumorous** and **non-tumorous**. The process involves several stages: **pre-training**, **fine-tuning**, and **unfreezing layers** for additional improvement. Here’s an in-depth explanation of the key components and decisions made during model development.

---

### **Why VGG19?**
- **VGG19** is a popular pre-trained convolutional neural network (CNN) known for its simplicity and effectiveness, particularly in image classification tasks.
- **Pre-trained** models like VGG19 are useful because they have already learned low-level and high-level features from a vast dataset (like **ImageNet**), which can be fine-tuned to suit specific tasks (like **tumor detection** in this case). 
- Using a pre-trained model allows us to avoid training the model from scratch, saving both time and computational resources.

---

### **Model Structure and Layers**

1. **Base Model: VGG19**

   The VGG19 model is used as a base, with the top classification layers (fully connected layers) removed to treat it as a **feature extractor**. This helps leverage its pre-learned weights for extracting meaningful features from images. The layers of VGG19 consist of:
   - **Convolutional layers**: These layers automatically detect basic features (edges, textures, shapes) in the images.
   - **MaxPooling layers**: These layers reduce the spatial dimensions, helping to keep the important features while reducing computational load.
   - **Fully connected layers (top layers)**: These layers, which were removed in our case, perform the final classification based on the extracted features.

   **Why freeze the layers of the base model initially?**

   When you initially use the pre-trained VGG19 model, we freeze the layers so that the weights in the convolutional base remain unchanged during the first phase of training. This prevents the model from unlearning the features it has already learned from the large dataset (ImageNet) and allows us to focus training on the new, added dense layers.

2. **Custom Classifier**

   After the base VGG19 model (without the top layer), we add custom fully connected layers to perform classification:
   - **Flatten Layer**: Converts the 2D feature maps from the VGG19 output into a 1D vector that can be passed to fully connected layers.
   - **Dense Layers**: These are traditional fully connected layers for classification. We added two dense layers (with 4608 and 1152 neurons, respectively), followed by a **Dropout** layer to reduce overfitting.
   - **Output Layer**: The output layer uses a **softmax** activation function to output probabilities for each class ("tumorous" or "non-tumorous").
---

### **Training the Model**

1. **Compiling the Model**:
   The model is compiled with the **categorical cross-entropy loss function** and **SGD optimizer** (Stochastic Gradient Descent) with a small learning rate (`0.0001`). This is because SGD is known to work well in fine-tuning pre-trained networks.

   - **Categorical Cross-Entropy**: Ideal for multi-class classification, which is the type of problem here, even though we have only two classes. It computes the loss by comparing the true labels with the predicted probabilities.
   - **SGD**: The optimizer used for training the model. It’s suitable for large networks and ensures that the model converges.

2. **Early Stopping, Checkpoints, and Learning Rate Reduction**:
   - **EarlyStopping**: Stops training when the validation loss stops improving, avoiding unnecessary epochs.
   - **ModelCheckpoint**: Saves the best model during training based on validation loss.
   - **ReduceLROnPlateau**: Reduces the learning rate when the validation accuracy stops improving to avoid overshooting the optimal weights.

   **Why these callbacks?**
   - **EarlyStopping** and **ModelCheckpoint**: These ensure that we don’t overfit and that we always retain the best-performing model.
   - **ReduceLROnPlateau**: By reducing the learning rate when improvement slows down, this allows for finer updates to the weights.

---

### **Fine-Tuning the Model**

The key to improving performance beyond what the pre-trained VGG19 model can achieve is **fine-tuning**. Fine-tuning involves unfreezing some of the layers in the pre-trained base model and allowing the weights in those layers to be updated during training. 

1. **Unfreezing Specific Layers**: 
   
   After initial training, we start fine-tuning the last layers of the VGG19 base model, allowing the model to adapt better to the specifics of our dataset (tumorous vs. non-tumorous images). In the first fine-tuning stage, we unfreeze the last two convolutional layers (`block5_conv3` and `block5_conv4`) to allow the model to learn more specific features.

   **Why unfreeze only some layers?**
   - The lower layers of the VGG19 model capture general features like edges and textures, which are applicable to many tasks. Freezing these layers prevents catastrophic forgetting.
   - The deeper layers (`block5_conv3` and `block5_conv4`) are more task-specific. Fine-tuning these layers allows the model to learn more domain-specific features for detecting tumors.

2. **Fine-Tuned Model: `vgg19_model_02.weights.h5`**:
   
   After fine-tuning, the model is saved as `vgg19_model_02.weights.h5`. These weights represent the model after unfreezing the last two convolutional layers and retraining.

---

### **Unfreezing All Layers**

In the final stage, we decide to **unfreeze all the layers** in the base model to allow full training of the VGG19 architecture. This means the entire network will be trained on our specific dataset, allowing the model to adapt fully to the task at hand.

1. **Unfreezing the Entire Model**:
   
   Here, we unfreeze all layers of VGG19 and retrain the entire network. This allows the model to learn more intricate patterns and adapt better to the new task.

   **Why unfreeze the entire model?**
   - Unfreezing the entire model ensures that the network is fully optimized for the task, learning both low-level and high-level features from the dataset.

2. **Final Model: `vgg_unfrozen.weights.h5`**:
   
   After fine-tuning all layers, the final model is saved as `vgg_unfrozen.weights.h5`, indicating that all layers were unfrozen and the model was fully retrained.

---

### **Model Evaluation**

Once the model is trained and fine-tuned, it is evaluated on both the **validation** and **test** datasets to assess its performance.
The evaluation gives us insights into how well the model generalizes to new, unseen data.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
