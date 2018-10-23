#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def main(argv):
    print("Convolutional Neural Network")
    from numpy import array,stack
    from pprint import pprint
    from tensorflow_utils import coord2d,image_pipeline,discriminator
    from matplotlib.pyplot import imshow,figure,show

    files=["gen_images/hc10_101.png","gen_images/hc10_102.png"]

    image,init_image_op=image_pipeline({'images':files,'labels':[1,2]})

    #Z=tf.placeholder(tf.float64,[None,2,2,1])

    #new_tensor=coord2d(Z,name='coord2d')

    #m=array([[[[.5],[.5]],[[.2],[.2]]]],dtype='float64')
    #mm=stack([*m,*m,*m,*m])

    #print(m.shape)
    #print(mm.shape)

    co=coord2d(((image['images']/255.0*-2.0)+1.0),name='acoord2d')

    d=discriminator(co)
    d.define_loss(image['labels'])

    with tf.Session() as session:
        print("Run")
        #a=session.run(new_tensor,feed_dict={Z: m})
        #pprint(a)

        session.run(init_image_op)

        #img=session.run(image)
        img=session.run(co)

        #la=img['labels']
        #img=img['images']

        

        #print(la)
        print(img.shape)
        pprint(img)
        #img=session.run(tf.image.grayscale_to_rgb(img))

        #pprint(img)
        #print(img.shape)

        figure()
        imshow(img[0])
        show()

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
