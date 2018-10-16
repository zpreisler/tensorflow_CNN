#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def main(argv):
    print("Convolutional Neural Network")
    import tensorflow as tf
    from model.data_pipeline import data_pipeline,get_labels
    from model.data_plot import plot_images,plot_softmax,save_softmax
    from model.model import discriminator
    from glob import glob

    files=glob('/home/zdenek/Projects/tensorflow_work/solid_images2/b*.conf')
    next_element,init_op=data_pipeline(files,batch=128)

    print(len(files))

    #inputs=tf.placeholder(tf.float64,[None,96,96,3])
    #outputs=tf.placeholder(tf.float64,[10])

    fl=discriminator(inputs=next_element['images'],outputs=next_element['labels']) 

    with tf.Session() as session:
        print("Start Session")
        """Init"""
        tf.global_variables_initializer().run(session=session)
        session.run(init_op)

        #a=session.run(next_element)
        #plot_images(a['images'],a['labels'])

        from matplotlib.pyplot import figure,show,plot
        count=0

        for i in range(2):
            true,softmax,acc,l,_=session.run([fl.outputs,fl.softmax,fl.accuracy,fl.loss,fl.train],feed_dict={fl.rate: 1e-3})
            print(i,acc,l)

            name='figures/f_%d.png'%count
            count+=1

            save_softmax(name,true,softmax)



if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
