##########################################################################
# model definitions. please don't edit this code, it is meant for model  #
# distribution only, via hacky github method. the model code in this     #
# file reflects the state of the trained model binaries in the repo in   #
# which you found this file.                                             #
##########################################################################


import tensorflow as tf




def kp_model(inputs):
    pad='same'
    x = inputs
    for i in range(2):
        x = tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,padding=pad)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out1 = x
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
    
    for i in range(2):
        x = tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,padding=pad)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out2 = x
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)

    for i in range(2):
        x = tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,padding=pad)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out3 = x
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)

    for i in range(2):
        x = tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,padding=pad)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
    x = tf.keras.layers.Concatenate()([x,out3])
    for i in range(2):
        x = tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,padding=pad)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
    x = tf.keras.layers.Concatenate()([x,out2])
    for i in range(2):
        x = tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,padding=pad)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
    x = tf.keras.layers.Concatenate()([x,out1])
    for i in range(2):
        x = tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,padding=pad)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,padding=pad)(x)
    x = tf.keras.layers.Conv2D(1,(3,3),activation=tf.nn.relu,padding=pad)(x)

    x = tf.math.subtract(tf.ones_like(x), tf.math.exp(tf.math.negative(x)))
    kp_activation = x
    return tf.keras.Model(inputs=inputs,outputs=kp_activation)

