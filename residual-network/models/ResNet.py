import tensorflow as tf
import tensorflow.keras.layers as layers

class BasicBlock(tf.keras.Model):
    expansion=1

    def __init__(self, channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        data_format, axis = 'channels_last', 3 # for Conv2D & BN

        self.conv1 = layers.Conv2D(channel, kernel_size=3, strides=stride, padding='same', use_bias=False, data_format=data_format)
        self.bn1 = layers.BatchNormalization(axis=axis)
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(channel, kernel_size=3, padding='same', use_bias=False, data_format=data_format)
        self.bn2 = layers.BatchNormalization(axis=axis)
        self.relu2 = layers.ReLU()

        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        out = self.relu2(out)

        return out

class Bottleneck(tf.keras.Model):
    expansion=4

    def __init__(self, channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        data_format, axis = 'channels_last', 3 # for Conv2D & BN

        self.stride=stride
        self.conv1 = layers.Conv2D(channel, kernel_size=1, padding='same', use_bias=False, data_format=data_format)
        self.bn1 = layers.BatchNormalization(axis=axis)
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(channel, kernel_size=3, strides=stride, padding='same', use_bias=False, data_format=data_format)
        self.bn2 = layers.BatchNormalization(axis=axis)
        self.relu2 = layers.ReLU()

        self.conv3 = layers.Conv2D(channel*expansion, kernel_size=1, padding='same', use_bias=False, data_format=data_format)
        self.bn3 = layers.BatchNormalization(axis=axis)
        self.relu3 = layers.ReLU()

        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)


        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))

        out += residual
        out = self.relu3(out)

        return out

class ResNet(tf.keras.Model):
    def __init__(self, block, layer):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.data_format, self.axis = 'channels_last', 3

        self.conv1 = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False, data_format=self.data_format)
        self.bn1 = layers.BatchNormalization(axis=self.axis)
        self.relu1 = layers.ReLU()
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same', data_format=self.data_format)

        self.layer1 = self._make_layer(block, 64, layer[0])
        self.layer2 = self._make_layer(block, 128, layer[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer[3], stride=2)

        self.reshape = layers.Reshape((512 * block.expansion, ))
        self.fc = layers.Dense(10, use_bias=False)
        self.softmax = layers.Softmax()

        self.size = [
            64,
            64 * block.expansion,
            128 * block.expansion,
            256 * block.expansion,
            512 * block.expansion
        ]

        #self.init_params()

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample=None

        if stride != 1 or self.inplanes != channel * block.expansion:
            downsample = tf.keras.Sequential()
            downsample.add(layers.Conv2D(channel * block.expansion, kernel_size=1, strides=stride, use_bias=False, data_format=self.data_format))
            downsample.add(layers.BatchNormalization(axis=self.axis))

        layer = tf.keras.Sequential()
        layer.add(block(channel, stride, downsample))

        for i in range(1, blocks):
            layer.add(block(channel))

        return layer

    def call(self, x):
        C1 = self.maxpool(self.relu1(self.bn1(self.conv1(x))))
        C2 = self.layer1(C1)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)

        out = self.reshape(C5)
        out = self.fc(out)
        out = self.softmax(out)

        return out

    def get_size(self):
        return self.size

def resnet18(pretrained=False):
    model = ResNet(BasicBlock, [2,2,2,2])
    return model

def resnet(net_size):
    if net_size == 18: return resnet18()
    else:
        raise NotImplementedError
