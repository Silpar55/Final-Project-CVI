from utils import *
from sklearn.model_selection import train_test_split

# Import Dataset
dataset_path = 'C:/ComputerVision/'
data = importDataSet(dataset_path)

# Balancing our data by analyzing and cutting off the histogram
data = balanceData(data, display=False)

# Preparing the data for processing
# Received a list of the images pathname and their respective steering in float
images_path, steerings = loadData(dataset_path, data)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(images_path, steerings, test_size=0.2)

print('Training data shape:', X_train.shape[0])
print('Test data shape:', X_test.shape[0])

# Preprocessing

# Creating the CNN model
model = create_model()
model.summary()

# Train our model
batch_size = 200
model.fit(
    batch_generator(X_train, y_train, batch_size, train_flag=True),
    steps_per_epoch=len(X_train)//batch_size,
    epochs=5,
    validation_data=batch_generator(X_test, y_test, batch_size, train_flag=False),
    validation_steps=len(X_test)//batch_size
)

model.save('model.h5')
print('Model saved')

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.legend(['Train', 'Validation'])
plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()