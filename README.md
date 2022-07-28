Face Emotion Recognition

We will build a deep learning model to classify facial expressions from images. Then we map the classified emotion to an emoji.

Using OpenCV's Har Cascade xml we are getting the bounding box of the faces in the webcam. Then we feed these boxes to the trained model for classification.

We have trained our model on the FER2013 dataset from Kaggle.

We have built a convolution neural network(CNN) to recognize facial emotions. 

Then we are mapping those emotions we detected with the corresponding emoji.


  ![20220727_174845](https://user-images.githubusercontent.com/83698103/181245241-9a2a3a6f-8a48-4fc8-88ed-27539802672c.jpg)

  
  
  ![Screenshot (265)](https://user-images.githubusercontent.com/83698103/181244876-acd5db5d-8a9e-43b9-81f6-9458feb2c276.png)
  

Result

Happy
![result11](https://user-images.githubusercontent.com/83698103/181291271-d58a5f2b-bda8-4c44-82e8-2551ad4b3784.jpeg)
Fearfull
![Screenshot (44)](https://user-images.githubusercontent.com/83698103/181433238-12834f65-18e0-40c6-9fa6-7cca465605b0.png)
Disgusting
![Screenshot (46)](https://user-images.githubusercontent.com/83698103/181433491-5e355a01-5398-45de-85ba-1e3937615b25.png)
Sad
![Screenshot (45)](https://user-images.githubusercontent.com/83698103/181433429-3c42b12a-38ee-460e-b41e-f8ad6f035a0b.png)
Angry
![Screenshot (43)](https://user-images.githubusercontent.com/83698103/181433591-fdf8130f-c297-4c43-9a2a-46f5a8042d73.png)
Surprised
![Screenshot (42)](https://user-images.githubusercontent.com/83698103/181433692-8935fe59-c48d-40c0-a157-23f4855df594.png)

Execution Video

https://user-images.githubusercontent.com/83698103/181296256-77f90553-ab95-45dc-b859-abdd2b52a560.mp4

