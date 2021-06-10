import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sn

print(tf.__version__)

df = pd.read_csv("./video_output/output_sum_63.csv")

df = df[1:]
print(df.tail())

for i in range(20):
    for j in range(3):
        print()
        df[str(3*i + j)] = df[str(3*i + j)] - df[str(j)]

size = df.shape[0]
print(size, "num of data")

df = df.sample(frac=1).reset_index(drop=True) #shuffle


test_ratio = 0.2
test_num = int(size - size * test_ratio)

train = df[:test_num]
test = df[test_num:]

col_name = [str(i) for i in range(0, 63)]
print(col_name)

print(train['FILENAME'].to_numpy())

train_x = train[col_name].to_numpy()
train_y = train['FILENAME'].to_numpy()
train_y = train_y.astype(np.int64)

# print(train_x)
# print(train_y)
# print(train_x.shape)
# print(len(train_y)) #1에서 14사이 정수 label

test_x = test[col_name].to_numpy()
test_y = test['FILENAME'].to_numpy()
test_y = test_y.astype(np.int64)

from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)

print(len(train_y), len(test_y)) #1에서 14사이 정수 label

model = keras.Sequential([
    keras.layers.Dense(63, activation = 'relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dropout(0.2),
    #keras.layers.Dense(30, activation = 'relu'),
    keras.layers.Dense(32, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
early_stop = EarlyStopping(monitor='val_loss', patience=7)
filename = 'model_save/my_model_63.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

hist = model.fit(train_x, train_y, epochs=100, batch_size=10, validation_data=(valid_x, valid_y),
                 callbacks=[early_stop, checkpoint])

test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

print('\n테스트 정확도:', test_acc)

prediction = model.predict(test_x[[5]])
print(prediction[0])
print(test_y[5])

from sklearn.metrics import confusion_matrix
# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

predictions = model.predict(test_x)
pred_y = (predictions > 0.5)
print(pred_y.shape)
print(test_y.shape)
matrix = confusion_matrix(test_y, pred_y.argmax(axis=1))
df_cm = pd.DataFrame(matrix, index = [i for i in "Xㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅢㅚㅟ"],
                  columns = [i for i in "Xㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅢㅚㅟ"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
#loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
#acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# model.save('model_save/my_model_21.h5')
# print('new model saved')
