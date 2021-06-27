import pandas as pd

class DataLoader:

    def __init__(self, data_path):
        self.full_data = pd.read_csv(data_path, index_col=0)
        self.product_data = pd.DataFrame()

    def filter_product(self,product_name):
        product_data = self.full_data[self.full_data['name']==product_name]
        product_data = product_data.dropna(subset=['comment'])
        return product_data

    def get_comments(self):
        comments = self.product_data['comment'].tolist()
        return comments

    def process_text(self,product_name):
        self.product_data = self.filter_product(product_name)
        comments = self.get_comments()
        return self.product_data, comments

if __name__ == "__main__":
    data = DataLoader('Data/bgg-15m-reviews.csv')
    product_data, comments = data.process_text('Pandemic')
    print('Done')
