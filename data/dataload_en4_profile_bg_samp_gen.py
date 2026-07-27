import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime, timedelta
import glob


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate(
            [idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]]


class GLORYDataset(Dataset):

    def __init__(self, glory_path, transform=None):
        """
        Args:
            glory_path (str): Path to GLORY data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.glory_files = self._get_files(glory_path)
        self.transform = transform

    def _get_files(self, path):
        """Retrieve all files from the given directory."""
        files = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                files.append(file_path)
        return sorted(files)

    def __len__(self):
        return len(self.glory_files)

    def __getitem__(self, idx, split='train'):
        glory_path = self.glory_files[idx]
        date_str = glory_path.split('/')[-1].split('_')[-2]
        date = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=7)
        formatted_date = date.strftime("%Y%m%d")

        if split == 'train':
            dir = "/mnt/data/zhishuai/Data/train_nw_pacific/mercatorglorys12v1_gl12_mean_"
        if split == 'test':
            dir = "/mnt/data/zhishuai/Data/test_nw_pacific/mercatorglorys12v1_gl12_mean_"

        path = dir + formatted_date
        glory_filename = glob.glob(f"{path}*")

        while len(glory_filename) != 1:

            idx = (idx + 1) % len(self.glory_files)
            glory_path = self.glory_files[idx]
            date_str = glory_path.split('/')[-1].split('_')[-2]
            date = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=7)
            formatted_date = date.strftime("%Y%m%d")

            path = dir + formatted_date
            glory_filename = glob.glob(f"{path}*")

        assert len(glory_filename) == 1
        glory_data = np.load(glory_filename[0])

        min_val = np.nanmin(glory_data, keepdims=True)
        max_val = np.nanmax(glory_data, keepdims=True)

        normalized_glory_data = (glory_data - min_val) / (max_val - min_val)
        normalized_glory_data[np.isnan(normalized_glory_data)] = 0

        gt = np.load(glory_path)
        normalized_gt = (gt - min_val) / (max_val - min_val)
        normalized_gt[np.isnan(normalized_gt)] = 0

        return normalized_glory_data, normalized_gt


class GLORYDatasetTest(Dataset):

    def __init__(self, glory_path, transform=None, split='test'):
        """
        Args:
            glory_path (str): Path to GLORY data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.glory_files = self._get_files(glory_path)
        self.transform = transform

        if split == 'train':
            dir = "/mnt/data/zhishuai/Data/train_nw_pacific/mercatorglorys12v1_gl12_mean_"
        if split == 'test':
            dir = "/mnt/data/zhishuai/Data/test_nw_pacific/mercatorglorys12v1_gl12_mean_"
        self.dir = dir

    def _get_files(self, path):
        """Retrieve all files from the given directory."""
        files = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path):
                files.append(file_path)
        return sorted(files)

    def __len__(self):
        return len(self.glory_files)

    def __getitem__(self, idx):
        glory_path = self.glory_files[idx]

        date_str = glory_path.split('/')[-1].split('_')[-2]
        date = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=7)
        formatted_date = date.strftime("%Y%m%d")

        path = self.dir + formatted_date
        glory_filename = glob.glob(f"{path}*")

        while len(glory_filename) != 1:
            # print(glory_filename)

            idx = (idx + 1) % len(self.glory_files)
            glory_path = self.glory_files[idx]
            date_str = glory_path.split('/')[-1].split('_')[-2]
            date = datetime.strptime(date_str, "%Y%m%d") - timedelta(days=7)
            formatted_date = date.strftime("%Y%m%d")

            path = self.dir + formatted_date
            glory_filename = glob.glob(f"{path}*")

        assert len(glory_filename) == 1
        glory_data = np.load(glory_filename[0])

        min_val = np.nanmin(glory_data, keepdims=True)
        max_val = np.nanmax(glory_data, keepdims=True)

        normalized_glory_data = (glory_data - min_val) / (max_val - min_val)

        # # zscore
        # img[:, n1:n2] -= sst_means
        # img[:, n1:n2] /= sst_stds

        normalized_glory_data[np.isnan(normalized_glory_data)] = 0

        gt = np.load(glory_path)
        normalized_gt = (gt - min_val) / (max_val - min_val)
        normalized_gt[np.isnan(normalized_gt)] = 0

        return normalized_glory_data, normalized_gt, min_val, max_val


def Denormalize(img, min_val, max_val):
    denormalized_image = img * (max_val - min_val) + min_val
    return denormalized_image


if __name__ == "__main__":
    # Example usage
    glory_path = "/public/home/yangnan/zhaomeihua/data/glory/train_nw_pacific/"

    dataset = GLORYDataset(glory_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in data_loader:
        normalized_glory_data, normalized_gt = batch
        print(normalized_glory_data.shape, normalized_gt.shape)
