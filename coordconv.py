#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def xfilter(dim):
    import tensorflow as tf
    x=tf.range(dim)

    x=x/(dim-1)
    x=(x*2)-1.0

    x=tf.expand_dims(x,0)
    x=tf.expand_dims(x,0)
    
    return x

def yfilter(dim):
    import tensorflow as tf
    y=tf.range(dim)

    y=y/(dim-1)
    y=(y*2)-1.0

    y=tf.reshape(y,(dim,1))
    y=tf.expand_dims(y,0)
    
    return y

class coord_conv2d:
    def __init__(self,
            #inputs,
            filters,
            kernel_size=1,
            strides=(1,1),
            padding='valid',
            name=None):
        self.kwargs={
                'filters': filters,
                'kernel_size': kernel_size,
                'strides': strides,
                'padding': padding,
                'name': name
                }

    def __call__(self,in_tensor):
        return tf.layers.conv2d(in_tensor,**self.kwargs)

def main(argv):
    print("Convolutional Neural Network")

    batch_size=tf.constant(1)
    xdim=tf.constant(3)
    ydim=tf.constant(3)

    x=xfilter(xdim)
    y=yfilter(ydim)

    x=tf.tile(x,[batch_size,ydim,1])
    y=tf.tile(y,[batch_size,1,xdim])

    x=tf.expand_dims(x,-1)
    y=tf.expand_dims(y,-1)

    z=tf.concat([x,y],-1)
    Z=tf.placeholder(tf.float64,[None,3,3,2])

    print(x)
    print(y)
    print(z)

    #v={'filters': 16, 'padding': 'valid'}

    c=coord_conv2d(filters=16)

    d=c(Z)

    with tf.Session() as session:
        print("Run")
        A=session.run(z)
        a=session.run(d,feed_dict={Z: A})

        from pprint import pprint
        pprint(a)

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
