# Meta Learning of heterogeneous PDE solutions

Partial Differential Equations (PDEs) are ubiquitous in the
physical sciences and in engineering, however, very rarely
these equations are solvable analytically. In most cases they
are very computationally demanding and require sophisti-
cated discretizations that render a universal algorithm inappli-
cable for different PDEs. In addition, even for a specific phys-
ical system described by a PDE, environmental and external
factors expressed as the boundary or the source function in a
PDE are constantly changing, in the traditional framework ,
for each such alteration , the PDE would need to be solved
from scratch. Using advances in meta-learning, we intro-
duce a Physics Informed Neural Network (PINN) approach to
tackle this important scenario. We verify our method on some
well known and fundamental PDEs and showcase how it can
be easily applied in any task. We further use insights from the
Neural Tangent Kernel to improve our algorithm yield better
solutions.


 We showcase our method on three problems of practical interest, the integration, the heat equation and the Poisson equation on 2 dimensions.

## Requirements
* [l2l](http://learn2learn.net/)
* [torchmeta](https://github.com/tristandeleu/pytorch-meta)
* [higher](https://github.com/facebookresearch/higher)

## Instructions

We provide pretrained models for each of the three problems. We provide examples in the examples directory that showcase how these models can be loaded and used on new tasks.

In case we need to retrain a model, we can execute train.py and choose the problem (the hyperparameters are the ones we provide, to change them we need to edit train.py).


```
python train.py --PDE integral
python train.py --PDE heat
python train.py --PDE poisson
```

 When executing the examples make sure to have the example.ipynb in the same directory as all other documents.


## To Do
* Include other PDEs, 3 dimensional and non linear.
* Use MAML extensions and PINN improvements.
* Compare time complexity during inference between numerical methods and our method in harder problems
* Compare our method with Neural Operators


