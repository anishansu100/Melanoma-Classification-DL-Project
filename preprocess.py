from keras.preprocessing.image import ImageDataGenerator

def get_data(file_path):
	# Initialising the generators for train and test data
	# The rescale parameter ensures the input range in [0, 1] 
	train_datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2)
	train_generator = train_datagen.flow_from_directory(
                  '/home/anish_pradhan/Melanoma-Classification-DL-Project/train',
                  target_size =(100, 100),  # target_size = input image size
				  color_mode="rgb", # for coloured images
                  batch_size = 500,
				  shuffle=True,
				  subset = 'training',
                  class_mode ='binary')
	test_generator = train_datagen.flow_from_directory(
                  '/home/anish_pradhan/Melanoma-Classification-DL-Project/train',
                  target_size =(100, 100),  # target_size = input image size
				  color_mode="rgb", # for coloured images
                  batch_size = 500,
				  shuffle=True,
				  subset = 'validation',
                  class_mode ='binary') 
	return train_generator, test_generator
	