import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class Mp3Dataset(Dataset):

    def __init__(self, csv_path, file_path)

        csvData = pd.read_csv(csv_path)
        self.file_names = []
        self.labels = []

        # go through the dataframe and import the filenames and genre labels
        for i in range(0, len(csvData)):
            self.
    




