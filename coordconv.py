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

def add_xy_coord(in_tensor,name=None):
    import tensorflow as tf
    with tf.name_scope(name,'add_xy_coord',[in_tensor]):
        batch_size=tf.shape(in_tensor)[0]
        xdim=in_tensor.shape[1]
        ydim=in_tensor.shape[2]

        x=xfilter(xdim)
        y=yfilter(ydim)

        x=tf.manip.tile(x,[batch_size,ydim,1])
        y=tf.manip.tile(y,[batch_size,1,xdim])

        x=tf.expand_dims(x,-1)
        y=tf.expand_dims(y,-1)

        x=tf.cast(x,dtype=in_tensor.dtype)
        y=tf.cast(y,dtype=in_tensor.dtype)

        return tf.concat([in_tensor,x,y],-1)

def main(argv):
    print("Convolutional Neural Network")
    from numpy import array,stack

    Z=tf.placeholder(tf.float64,[None,2,2,1])

    new_tensor=add_xy_coord(Z)

    m=array([[[[.5],[.5]],[[.2],[.2]]]],dtype='float64')
    mm=stack([*m,*m,*m,*m])

    print(m.shape)
    print(mm.shape)

    with tf.Session() as session:
        print("Run")
        a=session.run(new_tensor,feed_dict={Z: mm})
        from pprint import pprint
        pprint(a)

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
