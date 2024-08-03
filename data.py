from torchvision import datasets
import torchvision.transforms as transforms
import urllib

num_workers = 0
batch_size = 20
basepath = "./datasets"
transform = transforms.ToTensor()


def set_header_for(url, filename):
    opener = urllib.request.URLopener()
    opener.addheader(
        "User-Agent",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36",
    )
    opener.retrieve(url, f"{basepath}/{filename}")


set_header_for(
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train-images-idx3-ubyte.gz",
)
set_header_for(
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
)
set_header_for(
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
)
set_header_for(
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
train_data = datasets.MNIST(
    root="./datasets", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    root="./datasets", train=False, download=False, transform=transform
)
