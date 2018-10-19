class discriminator(object):
    def __init__(self,inputs,outputs,reuse=False,name='CNN'):
        import tensorflow as tf
        print('Initialize %s'%name)

        self.rate=tf.placeholder(tf.float64)

        self.inputs=inputs
        self.outputs=outputs

        self.build_graph(inputs,output_dim=outputs.shape[-1],reuse=reuse,name=name)

        self.define_loss(outputs)
        self.define_optimizer()
        self.define_training()

        self.prediction=tf.argmax(self.output_layer,1)
        self.true=tf.argmax(outputs,1)
        equality=tf.equal(self.prediction,self.true)
        self.accuracy=tf.reduce_mean(tf.cast(equality,tf.float64))
    
    def build_graph(self,inputs,output_dim,reuse,name):
        import tensorflow as tf
        with tf.variable_scope(name,reuse=reuse):
            self.conv_1=tf.layers.conv2d(
                    inputs=inputs,
                    filters=8,
                    kernel_size=[2,2],
                    padding='valid',
                    activation=tf.nn.leaky_relu)

            self.pool_1=tf.layers.average_pooling2d(
                    inputs=self.conv_1,
                    pool_size=[2,2],
                    strides=2)
            print(self.pool_1)

            self.conv_2=tf.layers.conv2d(
                    inputs=self.pool_1,
                    filters=16,
                    kernel_size=[2,2],
                    padding='valid',
                    activation=tf.nn.leaky_relu)

            self.pool_2=tf.layers.average_pooling2d(
                    inputs=self.conv_2,
                    pool_size=[2,2],
                    strides=2)
            print(self.pool_2)

            self.conv_3=tf.layers.conv2d(
                    inputs=self.pool_2,
                    filters=32,
                    kernel_size=[2,2],
                    padding='valid',
                    activation=tf.nn.leaky_relu)

            self.pool_3=tf.layers.average_pooling2d(
                    inputs=self.conv_3,
                    pool_size=[2,2],
                    strides=2)
            print(self.pool_3)

            n=self.pool_1.shape[1]*self.pool_1.shape[2]*self.pool_1.shape[3]
            self.flat_1=tf.reshape(self.pool_1,[-1,n])

            n=self.pool_2.shape[1]*self.pool_2.shape[2]*self.pool_2.shape[3]
            self.flat_2=tf.reshape(self.pool_2,[-1,n])

            n=self.pool_3.shape[1]*self.pool_3.shape[2]*self.pool_3.shape[3]
            self.flat_3=tf.reshape(self.pool_3,[-1,n])

            self.flat=tf.concat([self.flat_1,self.flat_2,self.flat_3],1)

            print(self.flat_2)
            print(self.flat_3)
            print(self.flat)

            self.dense_1=tf.layers.dense(
                    inputs=self.flat,
                    units=output_dim*output_dim*output_dim,
                    activation=tf.nn.leaky_relu)

            print(self.dense_1)

            self.dropout=tf.layers.dropout(
                    inputs=self.dense_1)

            self.dense_2=tf.layers.dense(
                    inputs=self.dropout,
                    units=output_dim,
                    activation=tf.nn.leaky_relu)

            print(self.dense_2)

            self.output_layer=self.dense_2

            self.softmax=tf.nn.softmax(self.dense_2)

    def define_loss(self,output_layer):
        import tensorflow as tf
        self.loss=tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=output_layer,
                    logits=self.output_layer
                    )
                )

    def define_optimizer(self):
        import tensorflow as tf
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.rate)

    def define_training(self):
        import tensorflow as tf
        self.train=self.optimizer.minimize(self.loss)

class detector(discriminator):
    def __init__(self,inputs,outputs,reuse=True,name='CNN'):
        import tensorflow as tf
        print('Initialize %s'%name)

        self.inputs=inputs
        self.outputs=outputs

        self.build_graph(inputs,output_dim=outputs.shape[-1],reuse=reuse,name=name)

    def build_graph(self,inputs,output_dim,reuse,name):
        import tensorflow as tf
        with tf.variable_scope(name,reuse=reuse):
            self.conv_1=tf.layers.conv2d(
                    inputs=inputs,
                    filters=8,
                    kernel_size=[2,2],
                    padding='valid',
                    activation=tf.nn.leaky_relu)

            self.pool_1=tf.layers.average_pooling2d(
                    inputs=self.conv_1,
                    pool_size=[2,2],
                    strides=2)
            print(self.pool_1)

            self.conv_2=tf.layers.conv2d(
                    inputs=self.pool_1,
                    filters=16,
                    kernel_size=[2,2],
                    padding='valid',
                    activation=tf.nn.leaky_relu)

            self.pool_2=tf.layers.average_pooling2d(
                    inputs=self.conv_2,
                    pool_size=[2,2],
                    strides=2)
            print(self.pool_2)

            self.conv_3=tf.layers.conv2d(
                    inputs=self.pool_2,
                    filters=32,
                    kernel_size=[2,2],
                    padding='valid',
                    activation=tf.nn.leaky_relu)

            self.pool_3=tf.layers.average_pooling2d(
                    inputs=self.conv_3,
                    pool_size=[2,2],
                    strides=2)
            print(self.pool_3)

            n=self.pool_1.shape[1]*self.pool_1.shape[2]*self.pool_1.shape[3]
            self.flat_1=tf.reshape(self.pool_1,[-1,n])

            n=self.pool_2.shape[1]*self.pool_2.shape[2]*self.pool_2.shape[3]
            self.flat_2=tf.reshape(self.pool_2,[-1,n])

            n=self.pool_3.shape[1]*self.pool_3.shape[2]*self.pool_3.shape[3]
            self.flat_3=tf.reshape(self.pool_3,[-1,n])

            self.flat=tf.concat([self.flat_1,self.flat_2,self.flat_3],1)

            print(self.flat_2)
            print(self.flat_3)
            print(self.flat)

            self.dense_1=tf.layers.dense(
                    inputs=self.flat,
                    units=output_dim*output_dim*output_dim,
                    activation=tf.nn.leaky_relu)
            print(self.dense_1)

            self.dense_2=tf.layers.dense(
                    inputs=self.dense_1,
                    units=output_dim,
                    activation=tf.nn.leaky_relu)
            print(self.dense_2)

            self.output_layer=self.dense_2
            self.softmax=tf.nn.softmax(self.dense_2)
