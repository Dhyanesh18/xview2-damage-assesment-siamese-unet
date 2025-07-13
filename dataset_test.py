from torch.utils.data import DataLoader
from dataset_class import XView2Dataset

dataset = XView2Dataset("train/")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for pre, post, mask in loader:
    print(pre.shape, post.shape, mask.shape)
    break
