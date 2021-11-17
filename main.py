import tensorflow as tf
import numpy as np
import glob
import statistics
import time

from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from scipy.spatial import distance

def define_encoder(input_shape=(64,64,1)):

	model = Sequential()
	model.add(Conv2D(20, kernel_size=5, activation='relu', kernel_initializer='he_uniform', use_bias=False, strides=(1,1), input_shape=input_shape))
	model.add(Conv2D(20, kernel_size=5, activation='relu', kernel_initializer='he_uniform', use_bias=False, strides=(2,2)))
	
	return model
	
def define_decoder():

	model = Sequential()
	model.add(Conv2D(20, kernel_size=5, activation='relu', kernel_initializer='he_uniform', use_bias=False, strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(5, activation='sigmoid'))
	
	return model
	
def define_model(encoder, decoder):

	model = Sequential()
	model.add(encoder)
	model.add(decoder)
	
	return model
	
batch_size = 64
epochs = 50
lr = 0.001
a = 0.0001

train_list = glob.glob('./dataset/grouped/train/*.png')
test_list = glob.glob('./dataset/grouped/test/*.png')
val_list = glob.glob('./dataset/grouped/val/*.png')

y_train = []
for i in train_list:
	res = i.split('_')[1]
	res = res.split('.')[0]
	l1 = tf.keras.utils.to_categorical(int(res[0])/2, 5)
	l2 = tf.keras.utils.to_categorical((int(res[1])-1)/2, 5)
	y_train.append([l1, l2])
y_train = np.array(y_train)

y_test = []
for i in test_list:
	res = i.split('_')[1]
	res = res.split('.')[0]
	l1 = tf.keras.utils.to_categorical(int(res[0])/2, 5)
	l2 = tf.keras.utils.to_categorical((int(res[1])-1)/2, 5)
	y_test.append([l1, l2])
y_test = np.array(y_test)
	
y_val = []
for i in val_list:
	res = i.split('_')[1]
	res = res.split('.')[0]
	l1 = tf.keras.utils.to_categorical(int(res[0])/2, 5)
	l2 = tf.keras.utils.to_categorical((int(res[1])-1)/2, 5)
	y_val.append([l1, l2])
y_val = np.array(y_val)

	

x_train = np.reshape(np.array([np.array(Image.open(fname)) for fname in train_list]), (16000,64,64,1))
x_test = np.reshape(np.array([np.array(Image.open(fname)) for fname in test_list]), (5000,64,64,1))
x_val = np.reshape(np.array([np.array(Image.open(fname)) for fname in val_list]), (4000,64,64,1))

print("Train shape: ", np.shape(x_train))
print("Test shape: ", np.shape(x_test))
print("Val shape: ", np.shape(x_val))


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)


encoder = define_encoder()
decoder_t1 = define_decoder()
decoder_t2 = define_decoder()
model_t1 = define_model(encoder, decoder_t1)
model_t2 = define_model(encoder, decoder_t2)

encoder.summary()
decoder_t1.summary()
decoder_t2.summary()
model_t1.summary()
model_t2.summary()

optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn1 = tf.keras.losses.CategoricalCrossentropy()
loss_fn2 = tf.keras.losses.CategoricalCrossentropy()

train_acc_metric_t1 = tf.keras.metrics.CategoricalAccuracy()
train_loss_metric_t1 = tf.keras.metrics.Mean()
train_acc_metric_t2 = tf.keras.metrics.CategoricalAccuracy()
train_loss_metric_t2 = tf.keras.metrics.Mean()

val_acc_metric_t1 = tf.keras.metrics.CategoricalAccuracy()
val_loss_metric_t1 = tf.keras.metrics.Mean()
val_acc_metric_t2 = tf.keras.metrics.CategoricalAccuracy()
val_loss_metric_t2 = tf.keras.metrics.Mean()

flatten = Flatten()

acc_list = []
loss_list = []
val_acc_list = []
val_loss_list = []
for epoch in range(epochs):
	
	start_time = time.time()
	
	for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

		with tf.GradientTape(persistent=True) as tape_enc:

			logits_t1 = model_t1(x_batch_train, training=True)
			logits_t2 = model_t2(x_batch_train, training=True)
			
			loss_t1 = loss_fn1(y_batch_train[:,0], logits_t1)
			loss_t2 = loss_fn2(y_batch_train[:,1], logits_t2)
			
			grads2_wrt_enc = tape_enc.gradient(loss_t2, encoder.trainable_weights)
			grads1_wrt_enc = tape_enc.gradient(loss_t1, encoder.trainable_weights)
			
			
			
			grads1_wrt_enc = [ flatten(np.reshape(grads1_wrt_enc[0], (1,np.shape(grads1_wrt_enc[0])[0], np.shape(grads1_wrt_enc[0])[1], np.shape(grads1_wrt_enc[0])[2], np.shape(grads1_wrt_enc[0])[3]) ) ), flatten(np.reshape(grads1_wrt_enc[1], (1,np.shape(grads1_wrt_enc[1])[0], np.shape(grads1_wrt_enc[1])[1], np.shape(grads1_wrt_enc[1])[2], np.shape(grads1_wrt_enc[1])[3]) ))]
			grads1_wrt_enc = tf.concat([grads1_wrt_enc[0], grads1_wrt_enc[1]], axis=1)
			
			grads2_wrt_enc = [ flatten(np.reshape(grads2_wrt_enc[0], (1,np.shape(grads2_wrt_enc[0])[0], np.shape(grads2_wrt_enc[0])[1], np.shape(grads2_wrt_enc[0])[2], np.shape(grads2_wrt_enc[0])[3]) ) ), flatten(np.reshape(grads2_wrt_enc[1], (1,np.shape(grads2_wrt_enc[1])[0], np.shape(grads2_wrt_enc[1])[1], np.shape(grads2_wrt_enc[1])[2], np.shape(grads2_wrt_enc[1])[3]) ))]
			grads2_wrt_enc = tf.concat([grads2_wrt_enc[0], grads2_wrt_enc[1]], axis=1)
			
			
			cos = distance.cosine(grads1_wrt_enc, grads2_wrt_enc)
			
			loss = loss_t1 + loss_t2 + a*cos*cos
	
		
		grad_t2 = tape_enc.gradient(loss, model_t2.trainable_weights)
		grad_t1 = tape_enc.gradient(loss, model_t1.trainable_weights)
		
		optimizer2.apply_gradients(zip(grad_t2, model_t2.trainable_weights))
		optimizer1.apply_gradients(zip(grad_t1, model_t1.trainable_weights))
		
	

		train_acc_metric_t1.update_state(y_batch_train[:,0], logits_t1)
		train_loss_metric_t1.update_state(loss_t1)
		train_acc_metric_t2.update_state(y_batch_train[:,1], logits_t2)
		train_loss_metric_t2.update_state(loss_t2)
		
	train_acc = [float(train_acc_metric_t1.result()), float(train_acc_metric_t2.result())]
	acc_list.append(train_acc)
	train_acc_metric_t1.reset_states()
	train_acc_metric_t2.reset_states()
	
	train_loss = [float(train_loss_metric_t1.result()), float(train_loss_metric_t2.result())]
	loss_list.append(train_loss)
	train_loss_metric_t1.reset_states()
	train_loss_metric_t2.reset_states()
	
	
	
	
	for x_batch_val, y_batch_val in val_dataset:
		
		val_logits_t1 = model_t1(x_batch_val, training=False)
		val_logits_t2 = model_t2(x_batch_val, training=False)
		
		loss_t1 = loss_fn1(y_batch_val[:,0], val_logits_t1)
		loss_t2 = loss_fn2(y_batch_val[:,1], val_logits_t2)
		
		val_acc_metric_t1.update_state(y_batch_val[:,0], val_logits_t1)
		val_loss_metric_t1.update_state(loss_t1)
		val_acc_metric_t2.update_state(y_batch_val[:,1], val_logits_t2)
		val_loss_metric_t2.update_state(loss_t2)
		

	val_acc = [float(val_acc_metric_t1.result()), float(val_acc_metric_t2.result())]
	val_acc_list.append(val_acc)
	val_acc_metric_t1.reset_states()
	val_acc_metric_t2.reset_states()
	
	val_loss = [float(val_loss_metric_t1.result()), float(val_loss_metric_t2.result())]
	val_loss_list.append(val_loss)
	val_loss_metric_t1.reset_states()
	val_loss_metric_t2.reset_states()
	
	

	
	

	print("TASK 1")
	print( "Epoch %d - %.2fs - loss: %.5f - accuracy: %.5f - val_loss: %.5f - val_accuracy: %.5f\n" % 
		(epoch,time.time() - start_time, train_loss[0], train_acc[0], val_loss[0], val_acc[0]) )

	print("TASK 2")
	print( "Epoch %d - %.2fs - loss: %.5f - accuracy: %.5f - val_loss: %.5f - val_accuracy: %.5f\n" % 
		(epoch,time.time() - start_time, train_loss[1], train_acc[1], val_loss[1], val_acc[1]) )


model_t2.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer2, metrics=['accuracy'])
model_t1.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer1, metrics=['accuracy'])


results_t2 = model_t2.evaluate(x_test, y_test[:,1], batch_size=64)
results_t1 = model_t1.evaluate(x_test, y_test[:,0], batch_size=64)

print("Task1 - test loss: ", results_t1[0], ", test acc: ", results_t1[1])
print("Task2 - test loss: ", results_t2[0], ", test acc: ", results_t2[1])




