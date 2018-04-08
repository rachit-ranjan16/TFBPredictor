
# ## Transcription Factors Binding Prediction
#
# ### Imports
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,SGD,Adam
import matplotlib.pyplot as plt


# #### Loading Data Files

train_data = pd.read_csv('data/train.csv').sample(frac=1)
test_data = pd.read_csv('data/test.csv')
sub = pd.read_csv('data/sampleSubmission.csv')


# #### Normalizing factor for encoding

norm = max([ord(i) for i in ['G','A','C','T']])


# ### Preprocessing
#  - Convert each character into ASCII code
#  - Convert 14 character DNA sequence into feature vector of ascii codes

def preprocess(data):
    """Converts the gene sequence into a feature list containing ASCII Values of the characters"""
    if data is None:
        return None
    out = []
    for gene in data:
        out.append([ord(i) for i in list(gene)])
    return out


# #### Split Training Data into Training and Validation
# #### Load Test Data

x_train, y_train = np.array(preprocess(train_data['sequence'][0:1400])).astype('float'), np.array(train_data['label'][0:1400]).astype('float')
x_val, y_val = np.array(preprocess(train_data['sequence'][1400:])).astype('float'), np.array(train_data['label'][1400:]).astype('float')
x_test = np.array(preprocess(test_data['sequence'])).astype('float')


# #### One Hot Encoding of Output Labels

num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)


# #### Normalize Training, Validation and Test Sets

x_train /= norm
x_val /= norm
x_test /= norm


# #### Plot Training Loss vs Validation Loss and Training Accuracy vs Validation Accuracy


def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)
    fig.savefig('Training and Validation Out.png')


# ### Deep Learning Model

# Accuracy 69% with Adam
pred_model = Sequential()
pred_model.add(Dense(10, activation='relu', input_shape=(14,)))
pred_model.add(Dropout(0.05))
pred_model.add(Dense(8, activation='relu'))
pred_model.add(Dense(6, activation='relu'))
pred_model.add(Dense(4, activation='relu'))
pred_model.add(Dense(2, activation='softmax'))
pred_model.summary()


# ### Set Hyperparameters and Train

learning_rate = .001
pred_model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(lr=learning_rate),
#                 optimizer=SGD(lr=learning_rate, momentum=0.2, decay=0.001, nesterov=True),
#                 optimizer=SGD(lr=learning_rate, momentum=0.02, decay=0.001, nesterov=False),
                  optimizer=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=True),
              metrics=['accuracy'])

batch_size = 32
epochs = 120

history = pred_model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=0,
    validation_data=(x_val, y_val))

plot_loss_accuracy(history)

score = pred_model.evaluate(x_val, y_val)
print("Accuracy %.6f" % score[1])


# ### Predictions on Test Set

out_list = []
submission = pd.DataFrame()
for out in pred_model.predict(x_test):
    out_list.append(out.argmax())
submission['id'] = test_data['id']
submission['prediction'] = out_list
submission.to_csv('submission.csv', index=False)
