#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
#class AddCoord2d(object):
#    def __init__(self,func):
#        self.x=None
#        self.y=None
#
#    def __call__(self,inputs,name=None,*args,**kwargs):
#        import tensorflow as tf
#        if not name:
#            self.name=name
#        with tf.name_scope(self.name):
#            shape=tf.shape(inputs)
#
#            self.x=tf.cast(self.xlayer(shape),
#                    dtype=inputs.dtype)
#            self.y=tf.cast(self.ylayer(shape),
#                    dtype=inputs.dtype)
#
#            return tf.concat([self.func(inputs,*args,**kwargs),self.x,self.y],-1)
#
#    def range(self,dim):
#        import tensorflow as tf
#        x=tf.range(dim)/(dim-1)
#        return (x*2)-1.0
#    
#    def xlayer(self,shape):
#        import tensorflow as tf
#        x=self.range(shape[1])
#        x=tf.expand_dims(x,0)
#        x=tf.expand_dims(x,0)
#        x=tf.manip.tile(x,[shape[0],shape[2],1])
#        x=tf.expand_dims(x,-1)
#        return x
#
#    def ylayer(self,shape):
#        import tensorflow as tf
#        y=self.range(shape[2])
#        y=tf.reshape(y,(shape[2],1))
#        y=tf.expand_dims(y,0)
#        y=tf.manip.tile(y,[shape[0],1,shape[1]])
#        y=tf.expand_dims(y,-1)
#        return y
#
def main(argv):
    print("Convolutional Neural Network")
    from numpy import array,stack
    from pprint import pprint
    from tensorflow_utils import coord2d

    Z=tf.placeholder(tf.float64,[None,2,2,1])

    new_tensor=coord2d(Z,name='coord2d')

    print(new_tensor.__dict__)

    m=array([[[[.5],[.5]],[[.2],[.2]]]],dtype='float64')
    mm=stack([*m,*m,*m,*m])

    print(m.shape)
    print(mm.shape)

    with tf.Session() as session:
        print("Run")
        a=session.run(new_tensor,feed_dict={Z: m})
        pprint(a)

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
