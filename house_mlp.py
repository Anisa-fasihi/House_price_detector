import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras import models, layers


EPOCHS = 200
BATCH_SIZE = 32

def load_house_data(inputPath):

    cols = ['bedroom', 'bathroom', 'area', 'zipcode', 'price']
    df = pd.read_csv(inputPath, sep=' ', header=None, names=cols)
    #print(df.shape)

    zipcodes = df["zipcode"].value_counts().keys().to_list()
    counts = df["zipcode"].value_counts().to_list()

    for (zipcode, count) in zip(zipcodes, counts):
        
        if count < 25:

            indx = df[df['zipcode'] == zipcode].index
            df.drop(indx, inplace=True)
    
    return df


def preprocessing_house_attributes(train, test):

    continuous = ['bedroom', 'bathroom', 'area']

    sc = StandardScaler()
    trainContinuous = sc.fit_transform(train[continuous])
    testContinuous = sc.transform(test[continuous])

    encoder = OneHotEncoder(sparse=False)
    trainCategorical = encoder.fit_transform(train[['zipcode']])
    # in one hot, we cannot have 1D array like train['zipcode']. we should have 2D array.
    testCategorical = encoder.transform(test[['zipcode']])

    train_x = np.hstack([trainContinuous, trainCategorical])
    test_x  = np.hstack([testContinuous, testCategorical])
    
    return train_x, test_x


def neural_network(x_train, x_test, y_train, y_test):

    net = models.Sequential([
                            layers.Dense(20, activation="relu"), 
                            layers.Dense(8, activation="relu"),
                            layers.Dense(1, activation="linear")])


    net.compile(optimizer="SGD", loss="mse")

    H = net.fit(x_train, y_train, batch_size= BATCH_SIZE, epochs= EPOCHS, validation_data= (x_test, y_test))

    loss = net.evaluate(x_test, y_test)
    print("loss = {:.2f}".format(loss))

    #net.save("mlp.h5")

    return net


df = load_house_data('HousesInfo.txt')
#print(df.shape)

train, test = train_test_split(df, test_size=0.2, random_state=42)

train_x, test_x = preprocessing_house_attributes(train, test)

maxPrice = train['price'].max()
train_y  = train['price']/maxPrice
test_y  = test['price']/maxPrice

model = neural_network(train_x, test_x, train_y, test_y)

pred = model.predict(test_x)
diff = pred.flatten() - test_y
percentDiff = (diff / test_y)*100
absPercentDiff = np.abs(percentDiff)
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
print("[INFO] mean error: {:.2f}%, std error: {:.2f}%".format(mean, std))
# or print(f"mean: {mean}, std: {std}")

# result: 
# sgd regressor: mean= 30, std= 40
# net: mean= 53, std= 79

