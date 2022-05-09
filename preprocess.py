from keras.preprocessing.image import ImageDataGenerator

def get_data(train_file_path, test_file_path, is_validation=False):
	"""
        Proprocess all the images to the right size in order to prepare for
		training and testing
        
        :param inputs: images, shape of (num_inputs, rows, cols, 3); during training, the shape is (batch_size, rows, cols, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes)
        """
	# Initialising the generators for train and test data
	if is_validation:
		# The rescale parameter ensures the input range in [0, 1] 
		train_datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2)
		train_generator = train_datagen.flow_from_directory(
					train_file_path,
					target_size =(200, 400),  # target_size = input image size
					color_mode="rgb", 
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
		# The rescale parameter ensures the input range in [0, 1] 
		train_datagen = ImageDataGenerator(rescale = 1./255)
		train_generator = train_datagen.flow_from_directory(
					train_file_path, # target_size = input image size
					color_mode="rgb",
					batch_size = 50,
					shuffle=True,
					class_mode ='binary')
		test_generator = train_datagen.flow_from_directory(
					test_file_path,
					color_mode="rgb",
					batch_size = 50,
					shuffle=True,
					class_mode ='binary') 
	return train_generator, test_generator
	