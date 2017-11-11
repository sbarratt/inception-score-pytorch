import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=False):
    """Computes the inception score of the generated images imgs

    imgs -- list of (HxWx3) numpy images normalized in the range [0,1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size to feed into inception
    """
    assert batch_size > 1

    if cuda:
        dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
    else:
        dtype = torch.FloatTensor

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=True).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x)

    # Get predictions
    N = len(imgs)
    imgs = [im.transpose(-1, 0, 1)[None, :] for im in imgs]
    preds = np.zeros((N, 1000))

    for i in range(N // batch_size + 1):
        start_batch = i*batch_size
        end_batch = (i+1)*batch_size
        if end_batch > N:
            end_batch = N

        imgs_batch = np.concatenate(imgs[start_batch:end_batch])
        imgs_batch = torch.Tensor(imgs_batch.astype(np.float32)).type(dtype)
        imgs_batch = Variable(imgs_batch)

        p_out = get_pred(imgs_batch)
        preds[start_batch:end_batch, :] = p_out.data.cpu().numpy()

    # Now compute the entropy
    py = np.mean(preds, axis=0)
    py /= np.sum(py)
    scores = []
    for i in range(preds.shape[0]):
        pyx = preds[i, :] / np.sum(preds[i, :])
        scores.append(entropy(pyx, py))

    return np.exp(np.mean(scores))

if __name__ == '__main__':
    print ("Generating images...")
    imgs = [np.random.uniform(0, 1, size=(64, 64, 3)).astype(np.float16) for _ in range(10)]
    print ("Generated Images")
    print (inception_score(imgs, cuda=True, batch_size=32, resize=True))
