import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.objaverse import ObjaverseDataset
from datasets.objaverse_skel import ObjaverseSkelDataset
from datasets.objaverse_hdf5 import ObjaverseHDF5Dataset
from datasets.objaverse_hdf5_coord import ObjaverseHDF5CoordDataset
from datasets.objaverse_hdf5_time import ObjaverseHDF5TimeDataset
from datasets.objaverse_skel_time import ObjaverseSkelTimeDataset
from datasets.objaverse_skel_graph import ObjaverseSkelGraphDataset
from datasets.omniobject3d import OmniObject3DDataset
from datasets.googlescan import GoogleScannedDataset

datasets = {
    'objaverse': ObjaverseDataset,
    'objaverse_skel': ObjaverseSkelDataset,
    'objaverse_hdf5': ObjaverseHDF5Dataset,
    'objaverse_hdf5_coord': ObjaverseHDF5CoordDataset,
    'objaverse_hdf5_time': ObjaverseHDF5TimeDataset,
    'objaverse_skel_time': ObjaverseSkelTimeDataset,
    'objaverse_skel_graph': ObjaverseSkelGraphDataset,
    'omniobject': OmniObject3DDataset,
    'gso': GoogleScannedDataset,
}

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self,
                 dataname = "", 
                 root_dir = "",
                 num_workers = 4,
                 batch_size = 1,
                 train = None,
                 validation = None,
                 test = None,
                 debug=False,
                 **kwargs):
        super().__init__()

        self.datafunc = datasets[dataname]
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.persistent_workers=True
        self.batch_size = batch_size
        self.debug = debug
        self.dataset_configs = dict()
        self.datasets = dict()

        if train is not None:
            self.dataset_configs["train"] = train
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.dataset_configs["test"] = validation
        if test is not None:
            self.dataset_configs["test"] = test

    def train_dataloader(self):

        self.datasets['train'] = self.datafunc(root_dir=self.root_dir, 
                                 cfg = self.dataset_configs["train"],
                                 debug = self.debug,
        )

        return DataLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        
    def val_dataloader(self):
        self.datasets['validation'] = self.datafunc(root_dir=self.root_dir,
                                cfg = self.dataset_configs["validation"],
                                debug = self.debug,
        )
        return DataLoader(self.datasets['validation'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def val_dataloader(self):
        self.datasets['test'] = self.datafunc(root_dir=self.root_dir,
                                cfg = self.dataset_configs["test"],
                                debug = self.debug,
        )
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)