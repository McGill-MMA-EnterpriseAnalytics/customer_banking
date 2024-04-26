import pandas as pd
from sklearn.model_selection import train_test_split

def create_test_dataset(full_data_path, test_data_path, test_size=0.1, random_state=123):
    df = pd.read_csv('/Users/sheidamajidi/Desktop/Winter2024/Winter2024-2/INSY695-076/Project/bank-full.csv', sep=';')
    
    # random testing set
    _, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    df_test.to_csv('/Users/sheidamajidi/Desktop/Winter2024/Winter2024-2/INSY695-076/Project/testing_set.csv', index=False)
    
    print("Test dataset saved to:", test_data_path)

create_test_dataset('/Users/sheidamajidi/Desktop/Winter2024/Winter2024-2/INSY695-076/Project/bank-full.csv', '/Users/sheidamajidi/Desktop/Winter2024/Winter2024-2/INSY695-076/Project/testing_set.csv')