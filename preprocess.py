from keras.preprocessing.image import ImageDataGenerator

def get_data(train_file_path, test_file_path, is_validation=False):
	# Initialising the generators for train and test data
	# The rescale parameter ensures the input range in [0, 1] 
	if is_validation:
		train_datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2)
		train_generator = train_datagen.flow_from_directory(
					train_file_path,
					target_size =(200, 400),  # target_size = input image size
					color_mode="rgb", # for coloured images
					batch_size = 50,
					shuffle=True,
					subset = 'training',
					class_mode ='binary')
		test_generator = train_datagen.flow_from_directory(
					train_file_path,
					target_size =(200, 400),  # target_size = input image size
					color_mode="rgb", # for coloured images
					batch_size = 50,
					shuffle=True,
					subset = 'validation',
					class_mode ='binary') 
	else:
		train_datagen = ImageDataGenerator(rescale = 1./255)
		train_generator = train_datagen.flow_from_directory(
					train_file_path, # target_size = input image size
					color_mode="rgb", # for coloured images
					batch_size = 50,
					shuffle=True,
					class_mode ='binary')
		test_generator = train_datagen.flow_from_directory(
					test_file_path,
					color_mode="rgb", # for coloured images
					batch_size = 50,
					shuffle=True,
					class_mode ='binary') 
	return train_generator, test_generator
	