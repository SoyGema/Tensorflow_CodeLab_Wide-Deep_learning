

![alt tag](https://github.com/SoyGema/Tensorflow_CodeLab_Wide-Deep_learning/blob/master/Tensor-GOT-Polymer/0_Logo.jpg)
# Tensorflow Codelab : dive into wide+deep learning 
Wide+Deep learning Neural Network Tensorflow
Data used : Game of thrones data set from kaggle

Made by : ssaaveda & SoyGema
Special thanks to : Baenans , Flipper83 and Laura Morillo 

You can find the codelab at https://soygema.github.io/TensorFlow-GOT/

You can directly run the codelad in a docker container 
```
docker pull tensorflow
docker run --rm -it -v $PWD:/model tensorflowgot/codelab --model_dir =/model
docker run --rm -it -v $PWD:/model -p 6006:6006 tensorflow/tensorflow tensorboard --logdir=/model
```

For running this codelab, you should have installed in your machine
--Python 2.7 or Python 3
      $ pip install python
--Sklearn
     $ pip insstall sklearn 
--Numpy 


This codelab has been updated in March 2017
Following upgrades :

0. Introduction
1. Visualization behind GoT dataset
2. Installing Tensorflow V.1
3. Codelab execution

You can actually read all the codelab without execute the code, although it is recommended for you to download and play with it. Take into account that chapter 1 only takes a first dig into the dataset with the objetive to know more about it and help distinguish in between continuous and categorical features. 

The code has been proved with Tensorflow v1. 

## CONFIGURATION
You should have configured before running the Codelab:
            # Python 
            $ sudo apt-get update
            $ sudo apt-get -y update
            $ python3 -V
            # Packages
To manage software packages for Python, you can type :
            $ pip3 install 'package_name'

## FAQ
1. Do I need to know a lot about Tensorflow for running into this codelab?
No, you should have a basic understanding of datascience, variable modeling and neural nets. The architecture is clearly defined 
2. Wich Neural Network Model are you using for the codelab?
We are using wide+deep learning model that combines the benefits of memorization+generalization 
3. Can I commit or colaborate?
Surething. We are looking for improvement in the following areas :
      A. Getting more accurate data. If you are passionate about Game of Thrones and want to give us a hand with the dataset, you are more than wellcome.
      B.Code . Yes, we would like to get more semantic and better with the code. Please don´t hesitate to push/commit in this journey
      C. Feedback. We have create a feedback form that you can find at the end of the codelab. Please feel free to colaborate.
      D. Share. Do you think anyone you meet would you like to learn more about Neuralnets and also loves Game of Throne? You can share this codelab .
4. Wich is the best way to experience this codelab?
You can experience this codelab in a lot of different ways. The most significant is to read chapter 1 and enjoy some datascience about game of thrones, then download the code, and change the model m, having the opportunity to change the optimizers and the hidden layers .
You can also read it without download the data .

Please, note that this is an going work and you might find some improvement areas.
Do not hesitate to give any idea or feedback about improvements!
Thanks!!

![alt tag](https://github.com/SoyGema/Tensorflow_CodeLab_Wide-Deep_learning/blob/master/Tensor-GOT-Polymer/3_Comic.png)

