import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
from log import Log
import os

class Training:
    def __init__(self):
        self.datasetPath = 'archive'
        self.imgShape = (64,64)
        self.log = Log()
    
    def getData(self,subset='training'):
        try:
            trainingData = tf.keras.utils.image_dataset_from_directory(
                self.datasetPath,
                validation_split=0.2,
                subset=subset,
                image_size= self.imgShape,
                seed = 58,
                batch_size = 32
            )
            msg = f"{subset.title()} dataset prepared successfully."
            self.log.logger(msg)
            return trainingData
        except Exception as e:
            self.log.logger(str(e))
    
    def visualizeData(self):
        try:
            msg = "Data visualization starts.."
            self.log.logger(msg)
            trainingData = self.getData()
            classNames = trainingData.class_names
            for images,lables in trainingData.take(1):
                plt.figure(figsize=(10,10))
                for i in range(18):
                    plt.subplot(6,3,i+1)
                    plt.imshow(images[i].numpy().astype('uint32'))
                    plt.title(classNames[lables[i].numpy().astype('uint8')])
        except Exception as e:
            self.log.logger(str(e))
                  
    def createModel(self):
        try:
            trainingData = self.getData();
            valData = self.getData(subset='validation')
            msg = "Model training start."
            self.log.logger(msg)
            LAYERS = [
                tf.keras.layers.Rescaling(1./255),
                tf.keras.layers.Conv2D(32,3,activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32,3,activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32,3,activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256,activation='relu'),
                tf.keras.layers.Dense(128,activation='relu'),
                tf.keras.layers.Dense(CLASSES)
            ]
            
            model = tf.keras.models.Sequential(LAYERS)
            model.compile(
                optimizer='adam',
                matrix=['accuracy'],
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logist=True)
            )
            
            hist=model.fit(
                trainingData,
                epochs=100,
                validation_data=valData
            )
            model.save("fruitLife.h5")
            self.log.logger("Model trained successfully.")
        except Exception as e:
            self.log.logger(str(e))
            
    def modelValidation(self):
        try:
            valData = self.getData(subset="validation")
            classNames = valData.class_names
            plt.figure(figsize=(10,30))
            model = tf.keras.models.load_model('fruitLife.h5')
            for images,lables in valData.take(1):
                for i in range(18):
                    plt.subplot(6,3,i+1)
                    img = np.reshape(images[i],(1,64,64,3))
                    probablity = model.predict(img)
                    yPred = classNames[np.argmax(probablity)]
                    yActual = classNames[lables[i].numpy().astype('uint32')]
                    plt.imshow(images[i].numpy().astype('uint8'))
                    plt.title(f"Model Evaluation")
                    plt.xlabel(f"Actual:{yActual}")
                    plt.ylabel(f"Predicted:{yPred}")
            self.log.logger("Model validated")
        except Exception as e:
            self.log.logger(str(e))
    
    def predict(self,path):
        try:
            img = tf.keras.preprocessing.image.load_img(path, target_size = self.imgShape)
            img1 = tf.keras.preprocessing.image.img_to_array(img)
            model = tf.keras.models.load_model("fruitLife.h5")
            y_pred=np.argmax(model.predict(img1.reshape((-1,64,64,3))))
            classes = ['Apple(1-5)','Apple(10-14)','Apple(5-10)','Banana(1-5)','Banana(10-15)','Banana(15-20)','Banana(5-10)',
                       'Carrot(1-2)','Carrot(3-4)','Expired','Tomato(1-5)','Tomato(10-15)','Tomato(5-10)','carrot(5-6)']
            classFruit = classes[y_pred]
            classFruit = classFruit.replace("("," ").replace(")","").split(" ")
            message = f"Predicted class is:{classFruit}"
            self.log.logger(message)
            return classFruit
        except Exception as e:
            self.log.logger(str(e))
                    
    def deleteFiles(self):
        try:
            for file in os.listdir("static/images"):
                filepath = os.path.join("static\\images",file)
                os.remove(filepath)
                self.log.logger(f"{filepath} removed successfully.")
        except Exception as e:
            self.log.logger(str(e))
                
              

    