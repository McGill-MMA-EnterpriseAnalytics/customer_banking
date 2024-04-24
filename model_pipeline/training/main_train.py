from train import train_model

def main_train():
    model = train_model('../Dataset/bank-full.csv')
    return model

if __name__ == '__main__':
    model = main_train()
    print(model)