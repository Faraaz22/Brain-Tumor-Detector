
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

### **Training and Model Fine-Tuning**

1. **Initial Training**:
   - The VGG19 model is first used as a feature extractor, with its layers frozen. Custom dense layers are added to classify the images.
   - The model is trained on the dataset for a few epochs and the weights are saved in `model_weights/`.

2. **Fine-Tuning**:
   - After the initial training, the last few layers of the VGG19 model are unfrozen to allow fine-tuning.
   - The model is retrained to improve performance, and the weights are updated accordingly.

3. **Saving the Model**:
   - After training or fine-tuning, the complete model is saved as `model.keras`. This file contains both the model architecture and the trained weights.

---

