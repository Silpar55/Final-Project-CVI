import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from sklearn.utils import shuffle


def getImgName(path):
	return path.split('/')[-1]


def importDataSet(path):
	columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
	dt = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)

	# Get only the name of the imageq
	dt['Center'] = dt['Center'].apply(getImgName)

	print('Total images imported: ', dt.shape[0])
	return dt


"""
	We need to balance our data to 0 since most of the time 
	the car does not steer in a certain angle.
	
	However, we don't want that our distribution between values is 
	largely different, therefore we need to cut off our redundant data
"""


def balanceData(data, display=True):
	n_bins = 31
	samples_per_bin = 3000

	hist, bins = np.histogram(data['Steering'], bins=n_bins)
	center = (bins[:-1] + bins[1:]) * 0.5

	if display:
		plt.bar(center, hist, width=0.06)
		plt.title("Steering Angle Histogram")
		plt.plot((-1, 1), (samples_per_bin, samples_per_bin))
		plt.show()

	remove_index_list = []
	for j in range(n_bins):
		bin_data_list = []
		for i in range(len(data['Steering'])):
			if bins[j] <= data['Steering'][i] <= bins[j + 1]:
				bin_data_list.append(i)
		bin_data_list = shuffle(bin_data_list)
		bin_data_list = bin_data_list[samples_per_bin:]
		remove_index_list.extend(bin_data_list)

	print('Removed Images:', len(remove_index_list))
	data.drop(data.index[remove_index_list], inplace=True)
	print('Remaining Data:', data.shape[0])

	if display:
		hist, bins = np.histogram(data['Steering'], bins=n_bins)
		plt.bar(center, hist, width=0.06)
		plt.title("Steering Angle Histogram with Removed Images")
		plt.plot((-1, 1), (samples_per_bin, samples_per_bin))
		plt.show()

	return data


def loadData(path, data):
	images_path = []
	steerings = []

	for i in range(len(data)):
		row = data.iloc[i]
		images_path.append(
			os.path.join(path, 'IMG', row['Center']))  # This return a list of ../dataset/IMG/filename.jpeg
		steerings.append(float(row['Steering']))

	images_path = np.asarray(images_path)
	steerings = np.asarray(steerings)
	return images_path, steerings


def augmentImage(image_path, steering):
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	rows, cols, _ = img.shape

	# Translation
	if np.random.rand() < 0.5:
		max_trans_pct = 0.2
		tx = np.random.uniform(-max_trans_pct, max_trans_pct) * cols
		ty = np.random.uniform(-max_trans_pct, max_trans_pct) * rows
		M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
		img = cv2.warpAffine(img, M_translate, (cols, rows))

	# Zoom
	if np.random.rand() < 0.5:
		scale = np.random.uniform(0.8, 1.2)
		# Resize the image
		img_zoomed = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

		# Crop or pad to keep original size
		zoom_rows, zoom_cols = img_zoomed.shape[:2]

		if scale < 1.0:
			# Pad the image to original size
			pad_vert = (rows - zoom_rows) // 2
			pad_horz = (cols - zoom_cols) // 2
			img = cv2.copyMakeBorder(
				img_zoomed,
				pad_vert,
				rows - zoom_rows - pad_vert,
				pad_horz,
				cols - zoom_cols - pad_horz,
				borderType=cv2.BORDER_REPLICATE
			)
		else:
			# Crop center to original size
			start_row = (zoom_rows - rows) // 2
			start_col = (zoom_cols - cols) // 2
			img = img_zoomed[start_row:start_row + rows, start_col:start_col + cols]

	if np.random.rand() < 0.5:
		# Brightness adjustment (multiply pixel values)
		brightness_factor = np.random.uniform(0.5, 1.5)
		img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)

	if np.random.rand() < 0.5:
		img = cv2.flip(img, 1)
		steering = -steering  # flip steering angle if applicable

	return img, steering


def preProcessing(img):
	img = img[60:135, :, :]  # Cropping
	img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # From RGB to YUV
	img = cv2.GaussianBlur(img, (5, 5), 0)  # Blurring
	img = cv2.resize(img, (200, 66))
	img = img / 255

	return img


def batch_generator(images_path, steerings, batch_size, train_flag=True):
	while True:
		img_batch = []
		steering_batch = []
		for i in range(batch_size):
			index = np.random.randint(0, len(images_path) - 1)
			if train_flag:
				img, steering = augmentImage(images_path[index], steerings[index])
			else:
				img = cv2.imread(images_path[index])
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				steering = steerings[index]

			img = preProcessing(img)
			img_batch.append(img)
			steering_batch.append(steering)
		yield np.asarray(img_batch), np.asarray(steering_batch)


def create_model():
	model = Sequential()
	model.add(Conv2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='relu'))
	model.add(Conv2D(36, (5, 5), (2, 2), activation='relu'))
	model.add(Conv2D(48, (5, 5), (2, 2), activation='relu'))
	model.add(Conv2D(64, (3, 3), (1, 1), activation='relu'))
	model.add(Conv2D(64, (3, 3), (1, 1), activation='relu'))

	model.add(Flatten())
	model.add(Dense(1164, activation='relu'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1))

	model.compile(
		optimizer=Adam(learning_rate=0.001),
		loss='mse', )

	return model
