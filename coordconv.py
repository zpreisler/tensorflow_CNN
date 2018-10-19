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

def main(argv):
    print("Convolutional Neural Network")

    batch_size=tf.constant(1)
    xdim=tf.constant(4)
    ydim=tf.constant(4)

    dtype=tf.float32

    x=xfilter(xdim)
    y=yfilter(ydim)

    print(x)
    print(y)

    x=tf.tile(x,[batch_size,ydim,1])
    y=tf.tile(y,[batch_size,1,xdim])

    with tf.Session() as session:
        print("Run")
        a,b=session.run([x,y])

        from pprint import pprint
        pprint(a)
        pprint(b)

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
