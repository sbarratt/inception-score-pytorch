import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=False):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [0,1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size to feed into inception
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

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
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        p_out = get_pred(batchv)
        preds[i*batch_size:i*batch_size + batch_size_i] = p_out.data.cpu().numpy()

    # Now compute the mean kl-div
    py = np.mean(preds, axis=0)
    scores = []
    for i in range(preds.shape[0]):
        pyx = preds[i, :]
        scores.append(entropy(pyx, py))
    mean_kl = np.mean(scores)

    return np.exp(mean_kl)

if __name__ == '__main__':
    print ("Generating images...")
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, imgs):
            self.imgs = imgs

        def __getitem__(self, index):
            return self.imgs[index]

        def __len__(self):
            return len(self.imgs)
    imgs = [np.random.uniform(0, 1, size=(3, 32, 32)).astype(np.float32) for _ in range(10)]
    imgs_dset = ImageDataset(imgs)

    print ("Calculating Inception Score...")
    print (inception_score(imgs_dset, cuda=True, batch_size=2, resize=True))
