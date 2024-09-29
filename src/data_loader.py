import os
import pandas as pd

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        dataset_file = os.path.join(self.data_path, 'dataset.csv')
        data = pd.read_csv(dataset_file)
        return data

    def save_processed_data(self, data):
        processed_file = os.path.join(self.data_path, 'processed_data.parquet')
        data.to_parquet(processed_file, index=False)

    def load_processed_data(self):
        processed_file = os.path.join(self.data_path, 'processed_data.parquet')
        if os.path.exists(processed_file):
            data = pd.read_parquet(processed_file)
            return data
        else:
            return None
