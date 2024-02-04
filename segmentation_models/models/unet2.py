from keras_applications import get_submodules_from_kwargs
import tensorflow as tf

from ._common_blocks import Conv2dBn
from ._utils import freeze_model, filter_keras_submodules
from ..backbones.backbones_factory import Backbones

backend = None
layers = None
models = None
keras_utils = None


# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)
        #print(x.shape)
        #print(input_tensor.shape)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def AttentionBlock(x, skip, i_filters):
    g1 = layers.Conv2D(i_filters,kernel_size = 1)(skip) 
    g1 = layers.BatchNormalization()(g1)
    x1 = layers.Conv2D(i_filters,kernel_size = 1)(x) 
    x1 = layers.BatchNormalization()(x1)

    g1_x1 = layers.Add()([g1,x1])
    psi = layers.Activation('relu')(g1_x1)
    psi = layers.Conv2D(1,kernel_size = 1)(psi) 
    psi = layers.BatchNormalization()(psi)
    psi = layers.Activation('sigmoid')(psi)
    x = layers.Multiply()([x, psi])
    return x


def DecoderTransposeX2Block_attention(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):
        print(f'Input tensor shape pre filtering= {input_tensor.shape}')
        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        print(f'Input tensor shape post filtering= {x.shape}')
        
        
        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)
            
        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:   
            print(f'Skip tensor shape  filtering= {skip.shape}')
            x = AttentionBlock(x, skip, filters)
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])
            
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):
        print(f'Input tensor shape = {input_tensor.shape}')
        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            print(f'Shape x = {x.shape}')
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer



def DecoderTransposeX2Block3D(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = -1 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):

        print(f'Input tensor shape = {input_tensor.shape}')
        x = tf.expand_dims(input_tensor, axis=0)
        x = layers.Conv3DTranspose(
            filters,
            kernel_size=(4, 4, 4),
            strides=(1, 2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(x)
        
        
        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)
        
        if skip is not None:
            print(f'Shape x = {x.shape}')
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, tf.expand_dims(skip, axis=0)])

        
        # x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)
        x1 = tf.keras.layers.Conv3D(
            filters,
            kernel_size=(1, 2, 2),
            #strides=(1, 2, 2),
            padding='same',
            data_format=None,
            dilation_rate=(1, 1, 1),
            groups=1,
            activation='relu',
            use_bias=True,
            kernel_initializer='he_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        ) (x)


        x2 = tf.keras.layers.Conv3D(
            filters,
            kernel_size=(1, 4, 4),
            #strides=(1, 2, 2),
            padding='same',
            data_format=None,
            dilation_rate=(1, 1, 1),
            groups=1,
            activation='relu',
            use_bias=True,
            kernel_initializer='he_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        ) (x)
        
        
        
        x3 = tf.keras.layers.Conv3D(
            filters,
            kernel_size=(1, 6, 6),
            # strides=(1, 3, 3),
            padding='same',
            data_format=None,
            dilation_rate=(1, 1, 1),
            groups=1,
            activation='relu',
            use_bias=True,
            kernel_initializer='he_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        ) (x)
        
        
        x = tf.add_n([x1, x2, x3])
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.squeeze(x, axis=0)   
        return x

    return layer


# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_unet2(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        #classes=1,
        #activation='sigmoid',
        use_batchnorm=True,
):
    input_ = backbone.input
    x = backbone.output[0]

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None
        print(i)
        # print(f'shape input first = {x.shape}')
        # print(f'N° of filters = {decoder_filters[i]}')
        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)
        
    print('ok')
    # model head (define number of output classes)
    '''
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        strides=(2,2),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)
    '''
    # create keras model instance
    model = models.Model(inputs=input_, outputs=(x))

    return model


# ---------------------------------------------------------------------
#  Unet Model
# ---------------------------------------------------------------------

def Unet2(
         backbone,
         encoder_features,
         #classes=1,
         #activation='sigmoid',
         encoder_freeze=False,
         decoder_block_type='upsampling',
         decoder_filters=(256, 128, 64, 32, 16),
         decoder_use_batchnorm=True,
         **kwargs
):
    """ Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        backbone: backbone
        input_shape: shape of input data/image ``(H, W, C)``, in general
            case you do not need to set ``H`` and ``W`` shapes, just pass ``(None, None, C)`` to make your model be
            able to process images af any size, but ``H`` and ``W`` of input images should be divisible by factor ``32``.
        classes: a number of classes for output (output shape - ``(h, w, classes)``).
        activation: name of one of ``keras.activations`` for last model layer
            (e.g. ``sigmoid``, ``softmax``, ``linear``).
        encoder_freeze: if ``True`` set all layers of encoder (backbone model) as non-trainable.
        encoder_features: a list of layer numbers or names starting from top of the model.
            Each of these layers will be concatenated with corresponding decoder block. 
        decoder_block_type: one of blocks with following layers structure:

            - `upsampling`:  ``UpSampling2D`` -> ``Conv2D`` -> ``Conv2D``
            - `transpose`:   ``Transpose2D`` -> ``Conv2D``

        decoder_filters: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.

    Returns:
        ``keras.models.Model``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """
    
    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)
    
    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    elif decoder_block_type == '3D':
        decoder_block = DecoderTransposeX2Block3D
    elif: decoder_block_type == 'transpose_att':
        decoder_block = DecoderTransposeX2Block_attention
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    model = build_unet2(
        backbone=backbone,
        decoder_block=decoder_block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        #classes=classes,
        #activation=activation,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
    )

    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    return model
