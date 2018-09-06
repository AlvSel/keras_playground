#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

data_path   = "./datasets/calculator_dataset.csv"
myDelimiter = ";"
outModel = "./models/my_naive_calculator.h5"

train_frac = 0.75
n_epochs = 5

def load_data(pDataset, pDelimiter):
	df      = pd.read_csv(data_path, header=None, sep=pDelimiter)
	train   = df.sample(frac=train_frac,random_state=200)
	test    = df.drop(train.index)
	x_train = train.iloc[:,:4]
	y_train = train.iloc[:,-1]
	x_test  = test.iloc[:,:4]
	y_test  = test.iloc[:,-1]
	return x_train,y_train,x_test,y_test


if __name__ == "__main__":
	print("Loading data...")
	x_train, y_train, x_test, y_test = load_data(data_path,myDelimiter)

	print("Build model...")
	model = Sequential()
	model.add(Dense(4, input_dim=4, activation='softmax'))
	model.add(Dense(2, activation='relu'))
	model.add(Dense(1, activation='relu'))
	model.compile(loss='mean_squared_error',
				  optimizer='adam', metrics=['accuracy'])

	print('Train...')
	model.fit(x_train, y_train,
	          epochs=n_epochs,
	          validation_data=(x_test, y_test))

	score, acc = model.evaluate(x_test, y_test)
	print('Test score:', score)
	print('Test accuracy:', acc)

	print(model.predict(x_test))

	model.save(outModel)
