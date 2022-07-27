Face Emotion Recognition

We will build a deep learning model to classify facial expressions from images. Then we map the classified emotion to an emoji.

Using OpenCV's Har Cascade xml we are getting the bounding box of the faces in the webcam. Then we feed these boxes to the trained model for classification.

We have trained our model on the FER2013 dataset from Kaggle.

We have built a convolution neural network(CNN) to recognize facial emotions. 

Then we are mapping those emotions we detected with the corresponding emoji.


  ![20220727_174845](https://user-images.githubusercontent.com/83698103/181245241-9a2a3a6f-8a48-4fc8-88ed-27539802672c.jpg)

  
  
  ![Screenshot (265)](https://user-images.githubusercontent.com/83698103/181244876-acd5db5d-8a9e-43b9-81f6-9458feb2c276.png)


LIBRARIES USED

Tensorflow

Numpy

Pillow

scipy

OpenCvPython

Tkinter


Steps:

1. Data Processing 

2. Image Augmentation

3.Feature Extraction

4.Training

5.Validation


![result](https://user-images.githubusercontent.com/83698103/181246114-995e2334-077f-4338-a894-f16a46fe172f.jpg)


