#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from collections import OrderedDict
from common.layers import Convolution, MaxPooling, ReLU, Affine, SoftmaxWithLoss, Flatten
from common.optimizer import RMSProp

class LeNet5:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv1_param={'filter_num':6, 'filter_size':5, 'pad':2, 'stride':1},
                 pool1_param={'pool_size':2, 'pad':0, 'stride':2},
                 conv2_param={'filter_num':16, 'filter_size':5, 'pad':0, 'stride':1},
                 pool2_param={'pool_size':2, 'pad':0, 'stride':2},
                 fc1_param={'input_size': 400, 'output_size': 120},
                 fc2_param={'input_size': 120, 'output_size': 84},
                 # fc3_param={'input_size': 84, 'output_size': 10},
                 fc3_param={'input_size': 84, 'output_size': 15},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        """
        input_size : tuple, 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
        conv_param : dict, 畳み込みの条件
        pool_param : dict, プーリングの条件
        hidden_size : int, 隠れ層のノード数
        output_size : int, 出力層のノード数
        weight_init_std ： float, 重みWを初期化する際に用いる標準偏差
        """
                
        filter1_num = conv1_param['filter_num']
        filter1_size = conv1_param['filter_size']
        filter1_pad = conv1_param['pad']
        filter1_stride = conv1_param['stride']
        
        pool1_size = pool1_param['pool_size']
        pool1_pad = pool1_param['pad']
        pool1_stride = pool1_param['stride']
        
        filter2_num = conv2_param['filter_num']
        filter2_size = conv2_param['filter_size']
        filter2_pad = conv2_param['pad']
        filter2_stride = conv2_param['stride']
        
        pool2_size = pool2_param['pool_size']
        pool2_pad = pool2_param['pad']
        pool2_stride = pool2_param['stride']
        
        input_size = input_dim[1]
#         print(f'input_size is : {input_size}')
        conv1_output_size = (input_size + 2*filter1_pad - filter1_size) // filter1_stride + 1 # 畳み込み後のサイズ(H,W共通)
#         print(f'conv1_output_size is: {conv1_output_size}')
        pool1_output_size = (conv1_output_size + 2*pool1_pad - pool1_size) // pool1_stride + 1 # プーリング後のサイズ(H,W共通)
#         print(f'pool1_output_size is: {pool1_output_size}')
        pool1_output_pixel = filter1_num * pool1_output_size * pool1_output_size # プーリング後のピクセル総数
#         print(f'pool1_output_pixel is: {pool1_output_pixel}')
        
        conv2_output_size = (pool1_output_size + 2*filter2_pad - filter2_size) // filter2_stride + 1 # 畳み込み後のサイズ(H,W共通)
#         print(f'conv2_output_size is: {conv2_output_size}')
        pool2_output_size = (conv2_output_size + 2*pool2_pad - pool2_size) // pool2_stride + 1 # プーリング後のサイズ(H,W共通)
#         print(f'pool2_output_size is: {pool2_output_size}')
        pool2_output_pixel = filter2_num * pool2_output_size * pool2_output_size # プーリング後のピクセル総数
#         print(f'pool2_output_pixel is: {pool2_output_pixel}')
        
        fc1_output_size = fc1_param['output_size']
        fc2_output_size = fc2_param['output_size']
        self.fc3_output_size = fc3_param['output_size']
        
        # 重みの初期化
        self.params = {}
        std = weight_init_std
        self.params['W1'] = np.sqrt(2.0 / (input_dim[0] * filter1_size * filter1_size)) * np.random.randn(filter1_num, input_dim[0], filter1_size, filter1_size) # W1は畳み込みフィルターの重みになる
        self.params['b1'] = np.zeros(filter1_num) #b1は畳み込みフィルターのバイアスになる
        
        self.params['W2'] = np.sqrt(2.0 / (filter1_num * filter2_size * filter2_size)) * np.random.randn(filter2_num, filter1_num, filter2_size, filter2_size) # W2は畳み込みフィルターの重みになる
        self.params['b2'] = np.zeros(filter2_num) #b2は畳み込みフィルターのバイアスになる
        
        self.params['W3'] = np.sqrt(2.0 / pool2_output_pixel) *  np.random.randn(pool2_output_pixel, fc1_output_size)
        self.params['b3'] = np.zeros(fc1_output_size)
        
        self.params['W4'] = np.sqrt(2.0 / fc1_output_size) *  np.random.randn(fc1_output_size, fc2_output_size)
        self.params['b4'] = np.zeros(fc2_output_size)
        
        self.params['W5'] = np.sqrt(2.0 / fc2_output_size) *  np.random.randn(fc2_output_size, self.fc3_output_size)
        self.params['b5'] = np.zeros(self.fc3_output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv1_param['stride'], conv1_param['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        self.layers['ReLU1'] = ReLU()
        self.layers['Pool1'] = MaxPooling(pool_h=pool1_size, pool_w=pool1_size, stride=pool1_stride, pad=pool1_pad)
        
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv2_param['stride'], conv2_param['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        self.layers['ReLU2'] = ReLU()
        self.layers['Pool2'] = MaxPooling(pool_h=pool2_size, pool_w=pool2_size, stride=pool2_stride, pad=pool2_pad)                            
        
        # TODO Flattenを定義する                            
#         self.layers['Flatten'] = Flatten() 
        
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['ReLU3'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['ReLU4'] = ReLU()                               
        self.layers['Affine3'] = Affine(self.params['W5'], self.params['b5'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def predict_label(self, x):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        return y

    def loss(self, x, t):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W5'], grads['b5'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        

        return grads

