import pandas as pd
import os
from CelebADataset import *
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../../datasets/celebA/list_attr_celeba.csv")
dir = "../../datasets/celebA/img_align_celeba/img_align_celeba"

trans = transforms.Compose([
  transforms.Resize((100, 100)),
  transforms.ToTensor()
])

training_data = CelebADataset(dir, df, trans, training=True)
test_data = CelebADataset(dir, df, trans, training=False)

img = training_data[np.random.randint(1, 150000)][0]
img = np.moveaxis(img.numpy(), 0, -1)

plt.imshow(img)
plt.show()
