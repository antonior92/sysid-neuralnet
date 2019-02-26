from torch.utils.data import DataLoader, Dataset






class DatasetExt(Dataset):

    @property
    def data_shape(self):
        raise Exception("Not implemented")

    @property
    def ny(self):
        return self.data_shape[1][0]  # first dimension of y, ie. # channels of y

    @property
    def nu(self):
        return self.data_shape[0][0] # first dimension of u, ie. # channels of u


class DataLoaderExt(DataLoader):
    @property
    def data_shape(self):
        """
            Returns the shape of the output
        """
        return self.dataset.data_shape

    @property
    def nu(self):
        return self.dataset.nu

    @property
    def ny(self):
        return self.dataset.ny
