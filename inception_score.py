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

    imgs -- list of (HxWx3) numpy images normalized in the range [0,1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size to feed into inception
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
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
    preds = np.zeros((N, 1000))

    # Wrap imgs in a dataset for simplicity
    imgs = [im.transpose(-1, 0, 1) for im in imgs]
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.imgs = imgs

        def __getitem__(self, index):
            return imgs[index]

        def __len__(self):
            return len(imgs)
    dataloader = torch.utils.data.DataLoader(ImageDataset(), batch_size=batch_size)

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        p_out = get_pred(batchv)
        preds[i*batch_size:i*batch_size + batch_size_i] = p_out.data.cpu().numpy()
    
    # Now compute the kl-div
    py = np.sum(preds, axis=0)
    scores = []
    for i in range(preds.shape[0]):
        pyx = preds[i, :]
        scores.append(entropy(pyx, py))

    return np.exp(np.mean(scores))

if __name__ == '__main__':
    print ("Generating images...")
    imgs = [np.random.uniform(0, 1, size=(32, 32, 3)).astype(np.float32) for _ in range(100)]
    print ("Generated Images")
    print (inception_score(imgs, cuda=True, batch_size=32, resize=True))
