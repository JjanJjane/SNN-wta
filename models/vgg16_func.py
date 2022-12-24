import tensorflow as tf

from tensorflow.python.keras.engine import training

#
#import lib_snn.layers
import lib_snn

from lib_snn.layers import tfn


#
# noinspection PyUnboundLocalVariable
# class VGG16_CIFAR(lib_snn.model.Model,tf.keras.layers.Layer):
# class VGG16(lib_snn.model.Model):
def VGG16(
        # def __init__(self, input_shape, data_format, conf):
        batch_size,
        input_shape,
        conf,
        model_name,
        include_top=True,
        # weights='imagenet',
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
        #nn_mode='ANN',
        #name='VGG16',
        dataset_name=None,
        **kwargs):


    data_format = conf.data_format


    #lib_snn.model.Model.__init__(self, input_shape, data_format, classes, conf)
    #lib_snn.model.Model.__init__(batch_size, input_shape, data_format, classes, conf)
    #Model = lib_snn.model.Model(batch_size, input_shape, data_format, classes, conf)


    #
    act_relu = 'relu'
    act_sm = 'softmax'

    #
    if conf.nn_mode=='ANN':
        dropout_conv_r = [0.2, 0.2, 0.0]      # DNN training
    elif conf.nn_mode=='SNN':
        #dropout_conv_r = [0.2, 0.2, 0.0]      # SNN training
        dropout_conv_r = [0.0, 0.0, 0.0]      # SNN training
    else:
        assert False

    #
    initial_channels = kwargs.pop('initial_channels', None)
    assert initial_channels is not None

    #
    use_bn_feat = conf.use_bn
    use_bn_cls = conf.use_bn

    #
    channels = initial_channels

    #
    if dataset_name=='ImageNet':
        n_dim_cls = 4096
    elif 'CIFAR' in dataset_name:
        n_dim_cls = 512
    else:
        assert False

    #
    k_init = 'glorot_uniform'
    #k_init = tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0,seed=None)




    #
    if conf.nn_mode=='ANN':
        act_type = 'relu'
        act_type_out = 'softmax'
    else:
        act_type = conf.n_type
        act_type_out = conf.n_type


    #
    tdbn_first_layer = conf.mode=='train' and conf.nn_mode=='SNN' and conf.input_spike_mode=='POISSON' and conf.tdbn
    tdbn = conf.mode=='train' and conf.nn_mode=='SNN' and conf.tdbn

    #

    if False:
        img_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)

        x = lib_snn.layers.InputGenLayer(name='in')(img_input)
        if conf.nn_mode=='SNN':
            x = tfn(lib_snn.activations.Activation(act_type=act_type,loc='IN',name='n_in'),x)

        #
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, en_tdbn=tdbn_first_layer, name='conv1')(x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat, kernel_initializer=k_init, name='conv1'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn_first_layer,name='bn_conv1'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv1'),x)
        x = tfn(tf.keras.layers.Dropout(dropout_conv_r[0], name='conv1_do'),x)
        #x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv1_1')(x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv1_1'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv1_1'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv1_1'),x)
        #x = tfn(lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv1_p'),x)
        #x = tfn(tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv1_p'),x)
        #x = tfn(lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv1_p', dynamic=True),x)
        x = tfn(lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv1_p'),x)

        #
        channels = channels * 2
        #x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv2'),x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv2'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv2'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv2'),x)
        x = tfn(tf.keras.layers.Dropout(dropout_conv_r[0], name='conv2_do'),x)
        #x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv2_1'),x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv2_1'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv2_1'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv2_1'),x)
        #x = tfn(lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv2_p'),x)
        #x = tfn(tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv2_p'),x)
        x = tfn(lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv2_p'),x)

        #
        channels = channels * 2
        #x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv3'),x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv3'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv3'),x)
        x = tfn(tf.keras.layers.Dropout(dropout_conv_r[1], name='conv3_do'),x)
        #x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv3_1'),x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv3_1'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3_1'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv3_1'),x)
        x = tfn(tf.keras.layers.Dropout(dropout_conv_r[1], name='conv3_1_do'),x)
        #x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv3_2'),x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv3_2'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3_2'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv3_2'),x)
        #x = tfn(lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv3_p'),x)
        #x = tfn(tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv3_p'),x)
        x = tfn(lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv3_p') ,x)

        #
        channels = channels * 2
        #x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv4'),x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv4'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv4'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv4'),x)
        x = tfn(tf.keras.layers.Dropout(dropout_conv_r[1], name='conv4_do'),x)
        #x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv4_1'),x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv4_1'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv4_1'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv4_1'),x)
        x = tfn(tf.keras.layers.Dropout(dropout_conv_r[1], name='conv4_1_do'),x)
        #x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv4_2'),x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv4_2'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv4_2'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv4_2'),x)
        #x = tfn(lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv4_p'),x)
        #x = tfn(tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv4_p'),x)
        x = tfn(lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv4_p') ,x)


        #
        #x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv5'),x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv5'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv5'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv5'),x)
        x = tfn(tf.keras.layers.Dropout(dropout_conv_r[1], name='conv5_do'),x)
        #x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv5_1'),x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv5_1'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv5_1'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv5_1'),x)
        x = tfn(tf.keras.layers.Dropout(dropout_conv_r[1], name='conv5_1_do'),x)
        #x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv5_2'),x)
        x = tfn(lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv5_2'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv5_2'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_conv5_2'),x)
        #x = tfn(lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv5_p'),x)
        #x = tfn(tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv5_p'),x)
        x = tfn(lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv5_p') ,x)


        #
        #if conf.nn_mode=='SNN' and not conf.snn.snn_training_spatial_first:

        #x = tfn(tf.keras.layers.Flatten(data_format=data_format),x)
        x = tfn(lib_snn.layers.Flatten(data_format=data_format),x)
        x = tfn(tf.keras.layers.Dropout(dropout_conv_r[2], name='flatten_do'),x)
        #x = tfn(lib_snn.layers.Dense(n_dim_cls, activation=act_relu, use_bn=use_bn_cls, kernel_initializer=k_init, name='fc1'),x)
        x = tfn(lib_snn.layers.Dense(n_dim_cls, kernel_initializer=k_init, name='fc1'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_fc1'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_fc1'),x)
        x = tfn(tf.keras.layers.Dropout(dropout_conv_r[2], name='fc1_do'),x)
        #x = tfn(lib_snn.layers.Dense(n_dim_cls, activation=act_relu, use_bn=use_bn_cls, kernel_initializer=k_init, name='fc2'),x)
        x = tfn(lib_snn.layers.Dense(n_dim_cls, kernel_initializer=k_init, name='fc2'),x)
        x = tfn(lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_fc2'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type,name='n_fc2'),x)
        x = tfn(tf.keras.layers.Dropout(dropout_conv_r[2], name='fc2_do'),x)
        #x = tfn(lib_snn.layers.Dense(classes, activation=act_sm, use_bn=False, last_layer=True, kernel_initializer=k_init, name='predictions'),x)
        x = tfn(lib_snn.layers.Dense(classes, last_layer=True, kernel_initializer=k_init, name='predictions'),x)
        x = tfn(lib_snn.activations.Activation(act_type=act_type_out,loc='OUT',name='n_predictions'),x)
        if conf.nn_mode=='SNN':
            x = tfn(lib_snn.activations.Activation(act_type='softmax',name='a_predictions'),x)

            if not conf.snn_training_spatial_first:
                out = x.read(conf.time_step-1)
            else:
                out = x
        else:
            out = x
    else:
        #
        img_input = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
        # x = img_input
        x = lib_snn.layers.InputGenLayer(name='in')(img_input)
        if conf.nn_mode=='SNN':
            x = lib_snn.activations.Activation(act_type=act_type,loc='IN',name='n_in')(x)

        #
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, en_tdbn=tdbn_first_layer, name='conv1')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', use_bn=use_bn_feat, kernel_initializer=k_init, name='conv1')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn_first_layer,name='bn_conv1')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv1')(x)
        x = tf.keras.layers.Dropout(dropout_conv_r[0], name='conv1_do')(x)
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv1_1')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv1_1')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv1_1')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv1_1')(x)
        #x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv1_p')(x)
        #x = tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv1_p')(x)
        #x = lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv1_p', dynamic=True)(x)
        x = lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv1_p')(x)

        #
        channels = channels * 2
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv2')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv2')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv2')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv2')(x)
        x = tf.keras.layers.Dropout(dropout_conv_r[0], name='conv2_do')(x)
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv2_1')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv2_1')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv2_1')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv2_1')(x)
        #x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv2_p')(x)
        #x = tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv2_p')(x)
        x = lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv2_p')(x)

        #
        channels = channels * 2
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv3')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv3')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv3')(x)
        x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv3_do')(x)
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv3_1')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv3_1')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3_1')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv3_1')(x)
        x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv3_1_do')(x)
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv3_2')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv3_2')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv3_2')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv3_2')(x)
        #x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv3_p')(x)
        #x = tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv3_p')(x)
        x = lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv3_p') (x)

        #
        channels = channels * 2
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv4')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv4')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv4')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv4')(x)
        x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv4_do')(x)
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv4_1')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv4_1')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv4_1')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv4_1')(x)
        x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv4_1_do')(x)
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv4_2')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv4_2')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv4_2')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv4_2')(x)
        #x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv4_p')(x)
        #x = tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv4_p')(x)
        x = lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv4_p') (x)


        #
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv5')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv5')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv5')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv5')(x)
        x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv5_do')(x)
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv5_1')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv5_1')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv5_1')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv5_1')(x)
        x = tf.keras.layers.Dropout(dropout_conv_r[1], name='conv5_1_do')(x)
        #x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', activation=act_relu, use_bn=use_bn_feat, kernel_initializer=k_init, name='conv5_2')(x)
        x = lib_snn.layers.Conv2D(channels, 3, padding='SAME', kernel_initializer=k_init, name='conv5_2')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_conv5_2')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_conv5_2')(x)
        #x = lib_snn.layers.MaxPool2D((2, 2), (2, 2), name='conv5_p')(x)
        #x = tf.keras.layers.AveragePooling2D((2, 2), (2, 2), name='conv5_p')(x)
        x = lib_snn.layers.AveragePooling2D((2, 2), (2, 2), name='conv5_p') (x)


        #
        x = tf.keras.layers.Flatten(data_format=data_format)(x)
        x = tf.keras.layers.Dropout(dropout_conv_r[2], name='flatten_do')(x)
        #x = lib_snn.layers.Dense(n_dim_cls, activation=act_relu, use_bn=use_bn_cls, kernel_initializer=k_init, name='fc1')(x)
        x = lib_snn.layers.Dense(n_dim_cls, kernel_initializer=k_init, name='fc1')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_fc1')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_fc1')(x)
        x = tf.keras.layers.Dropout(dropout_conv_r[2], name='fc1_do')(x)
        #x = lib_snn.layers.Dense(n_dim_cls, activation=act_relu, use_bn=use_bn_cls, kernel_initializer=k_init, name='fc2')(x)
        x = lib_snn.layers.Dense(n_dim_cls, kernel_initializer=k_init, name='fc2')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn,name='bn_fc2')(x)
        x = lib_snn.activations.Activation(act_type=act_type,name='n_fc2')(x)
        x = tf.keras.layers.Dropout(dropout_conv_r[2], name='fc2_do')(x)
        #x = lib_snn.layers.Dense(classes, activation=act_sm, use_bn=False, last_layer=True, kernel_initializer=k_init, name='predictions')(x)
        x = lib_snn.layers.Dense(classes, last_layer=True, kernel_initializer=k_init, name='predictions')(x)
        x = lib_snn.activations.Activation(act_type=act_type_out,loc='OUT',name='n_predictions')(x)
        if conf.nn_mode=='SNN':
            x = lib_snn.activations.Activation(act_type='softmax',name='a_predictions')(x)




    #model = training.Model(img_input, x, name=name)
    model = lib_snn.model.Model(img_input, x, batch_size, input_shape,  classes, conf, name=model_name)
    #model = lib_snn.model.Model(img_input, out, batch_size, input_shape,  classes, conf, name=model_name)
    #model = lib_snModel.init_graph(img_input, x, name=name)



    # return training.Model(img_input, x, name=self.name)

    #        self.out = x

    #    def build(self, input_shape):
    #        img_input = tf.keras.layers.Input(shape=self.in_shape, batch_size=self.batch_size)
    #        # create model
    #        self.model = training.Model(img_input, self.out, name=self.name)

    # self.model.load_weights(weights)

    # self.load_weights = weights

    # if weights is not None:
    # self.model.load_weights(weights)
    # self.set_weights(weights)
    # self.model.set_weights(weights)
    # print('load weights done')

    #model.summary()

    return model
