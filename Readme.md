# WGAN-tensorlayer

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

$$
x^2+2
$$