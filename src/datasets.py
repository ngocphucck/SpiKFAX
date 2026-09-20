from torchvision import datasets, transforms
import tonic
import torch


def get_datasets(dataset_name):
    if dataset_name == 'MNIST':
        data_path = '../data/data/static/MNIST'
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
        test_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,))])
        mnist_train = datasets.MNIST(data_path, train=True, download=False, transform=transform)
        mnist_test = datasets.MNIST(data_path, train=False, download=False, transform=test_transform)

        return mnist_train, mnist_test

    elif dataset_name == 'F-MNIST':
        data_path = '../data/data/static/FashionMNIST'
        transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.Grayscale(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,))])
        test_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0,), (1,))])
        mnist_train = datasets.FashionMNIST(data_path, train=True, download=False, transform=transform)
        mnist_test = datasets.FashionMNIST(data_path, train=False, download=False, transform=test_transform)

        return mnist_train, mnist_test

    elif dataset_name == 'CIFAR100':
        data_path = '../data/data/static/CIFAR100'
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        test_transform = transforms.Compose([
            transforms.ToTensor()])
        cifar100_train = datasets.CIFAR100(data_path, train=True, download=False, transform=transform)
        cifar100_test = datasets.CIFAR100(data_path, train=False, download=False, transform=test_transform)

        return cifar100_train, cifar100_test

    elif dataset_name == 'CIFAR10':
        data_path = '../data/data/static/CIFAR10'
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        test_transform = transforms.Compose([
            transforms.ToTensor()])
        cifar10_train = datasets.CIFAR10(data_path, train=True, download=False, transform=transform)
        cifar10_test = datasets.CIFAR10(data_path, train=False, download=False, transform=test_transform)

        return cifar10_train, cifar10_test

    elif dataset_name == 'N-MNIST':
        data_path = '../data/data/neuromorphic'
        
        # N-MNIST sensor size is strictly (34, 34, 2)
        sensor_size = tonic.datasets.NMNIST.sensor_size

        # ToFrame converts raw events into a dense [Time, Channels, H, W] tensor
        transform = tonic.transforms.Compose([
            tonic.transforms.Downsample(sensor_size=sensor_size, target_size=(32, 32)),
            tonic.transforms.ToFrame(sensor_size=(32, 32, 2), n_time_bins=10)
        ])
        
        nmnist_train = tonic.datasets.NMNIST(data_path, train=True, transform=transform)
        nmnist_test = tonic.datasets.NMNIST(data_path, train=False, transform=transform)

        return nmnist_train, nmnist_test
    
    elif dataset_name == 'CIFAR10-DVS':
        data_path = '../data/data/neuromorphic'
        
        # Original resolution is 128x128, downsample to 32x32 to match CIFAR-10 models
        sensor_size = tonic.datasets.CIFAR10DVS.sensor_size  # (128, 128, 2)
        
        transform = tonic.transforms.Compose([
            tonic.transforms.Downsample(sensor_size=sensor_size, target_size=(32, 32)),
            tonic.transforms.ToFrame(sensor_size=(32, 32, 2), n_time_bins=10)
        ])
        
        # CIFAR10-DVS doesn't have a built-in train/test split parameter in tonic, 
        # so it's typically split manually (e.g., 90/10) or loaded completely with a subset selector.
        dataset = tonic.datasets.CIFAR10DVS(data_path, transform=transform)
        
        # Splitting 90% train, 10% test (standard practice for CIFAR10-DVS)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        cifar10dvs_train, cifar10dvs_test = torch.utils.data.random_split(dataset, [train_size, test_size])

        # print('CIFAR10 dvs: ', cifar10dvs_train[0][0].shape)
        # print('Length of the dataset: ', len(cifar10dvs_train))
        # exit()

        return cifar10dvs_train, cifar10dvs_test

    elif dataset_name == 'DVS128-Gesture':
        data_path = '../data/data/neuromorphic'
        
        sensor_size = tonic.datasets.DVSGesture.sensor_size  # (128, 128, 2)
        
        # Downsample to 64x64 to balance model memory and spatial fidelity, 
        # and bin events into n_steps time frames
        transform = tonic.transforms.Compose([
            tonic.transforms.Downsample(sensor_size=sensor_size, target_size=(32, 32)),
            tonic.transforms.ToFrame(sensor_size=(32, 32, 2), n_time_bins=10)
        ])
        
        dvs_train = tonic.datasets.DVSGesture(data_path, train=True, transform=transform)
        dvs_test = tonic.datasets.DVSGesture(data_path, train=False, transform=transform)

        # print('DVS train: ', dvs_train[0])
        # print('Length of the dataset: ', len(dvs_train))
        # exit()

        return dvs_train, dvs_test