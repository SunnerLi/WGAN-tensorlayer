# WGAN-tensorlayer

[![Packagist](https://img.shields.io/badge/Tensorflow-1.3.0-yellow.svg)]()
[![Packagist](https://img.shields.io/badge/Tensorlayer-1.6.4-blue.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()
[![Packagist](https://img.shields.io/badge/OpenCV-3.1.0-brightgreen.svg)]()

![](https://github.com/SunnerLi/www/blob/master/img/curve_merge.png)

Abstraction
---
In this project, there're three contributions I have done. First, I merge the LSUN API and ensure the original function can work in python 3+ environment. Second, The WGAN and DCGAN are reproduced toward MNIST. At last, I try to do the experiments toward the disadvantage of WGAN. Unfortunately, **the idea cannot improve the pros of the architecture**. You can see the LSUN in the wrapper [folder](https://github.com/SunnerLi/www/tree/master/wrapper/lsun). After you run `train_mnist.py`, `train_lsun.py` or `train_celeba.py`, the corresponding training process will be started. The following section will introduce my experiments.     

Introduction
---
In GAN territories, the collapse problem and instability are the fatal issue. After a year, DCGAN was purposed which try to solve this problem. However, there're two problems: First, You should be very carefully to design the architecture, or it cannot converge; collapse problem still cannot be solved. As the result, WGAN was purpose in the early of 2017. It successfully solve the collapse problem (no problem until now). 

Experiment
---
However, after observing the WGAN process, three disadvantages that I try to solve. 
1. the speed of convergence is slow.     
2. The measure term (Earth-Mover distance) is a typical function which is sharp in the center of coordinate. This kinds of function might cause some issue (for example, it cannot derivative smoothly). Maybe the other smoother function can be adopted.    
3. With considering the gradient penalty, you should make the whole process go through discriminator for 3 times which is time consuming. A guessing is in my mind if we can reduce this process.     

![](https://github.com/SunnerLi/www/blob/master/img/curve1.jpg)

In original WGAN, the earth-mover distance is adopted to give the meaningful loss toward the training. The shape of earth-mover shows at the above image. On the other hand, I purpose the improvement curve which is shown at below. Through it goes through the non-linear transformation toward the original function curve, it still obey the definition of Lipschitz function.    

![](https://github.com/SunnerLi/www/blob/master/img/curve2.jpg)

This kind of function has three properties:
1. It's continuous at anywhere
2. It can restrict the revised value with upper limit of 1
3. The function form is similar to gaussian    

Result
---
I do the three experiments:
1. First, in WGAN training process, you should clip the weight to the specific section (-c, +c). This action is similar to regularize the weights. As the result, I try to remove the gradient penalty but just doing l2 normalization toward the weights. 
2. I implement the non-linear transformation toward gradient penalty term, and does the other work as origin. 
3. At last, I purpose an idea of merge. For the gradient penalty process, you should create the combination of noise and actual image. Next, you should make it go through the discriminator which costs the extra computation time. I simplified the combination term and just use the combination of origin batch data to compute the gradient directly.   

![](https://github.com/SunnerLi/www/blob/master/img/loss1.jpg)
![](https://github.com/SunnerLi/www/blob/master/img/loss2.jpg)
![](https://github.com/SunnerLi/www/blob/master/img/loss3.jpg)

Actually, they are whole failed. The above three image shows the loss curve toward each experiments. As you can see, they cannot converge eventually. And the following shows the image they generated. They cannot capture the distribution of real data. What's worse, there's some kinds of collapse problem.    

![](https://github.com/SunnerLi/www/blob/master/img/generate_result.jpg)