



from torch.utils.data import DataLoader, Dataset


class DatasetExt(Dataset):

    @property
    def data_shape(self):
        raise Exception("Not implemented")


class DataLoaderExt(DataLoader):

    @property
    def data_shape(self):
        """
            Returns the shape of the output
        """
        return self.dataset.data_shape



