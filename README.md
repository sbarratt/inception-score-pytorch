# Inception Score Pytorch

Pytorch was lacking code to calculate the Inception Score for GANs. This repository fills this gap.
However, we do not recommend using the Inception Score to evaluate generative models, see [our note](https://arxiv.org/abs/1801.01973) for why.

## Getting Started

Clone the repository and navigate to it:
```
$ git clone git@github.com:sbarratt/inception-score-pytorch.git
$ cd inception-score-pytorch
```

To generate random 64x64 images and calculate the inception score, do the following:
```
$ python inception_score.py
```

The only function is `inception_score`. It takes a list of numpy images normalized to the range [0,1] and a set of arguments and then calculates the inception score. Please assure your images are 299x299x3 and if not (e.g. your GAN was trained on CIFAR), pass `resize=True` to the function to have it automatically resize using bilinear interpolation before passing the images to the inception network.

```python
def inception_score(imgs, cuda=True, batch_size=32, resize=False):
    """Computes the inception score of the generated images imgs

    imgs -- list of (HxWx3) numpy images normalized in the range [0,1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size to feed into inception
    """
```

### Prerequisites

You will need [torch](http://pytorch.org/), [torchvision](https://github.com/pytorch/vision), [numpy/scipy](https://scipy.org/).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Inception Score from [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
