#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def main(argv):
    print("Convolutional Neural Network")
    import tensorflow as tf
    from model.data_pipeline import data_pipeline,get_labels
    from model.data_plot import plot_images,save_images,plot_softmax,save_softmax
    from model.model import discriminator
    from glob import glob

    files=glob('/home/zdenek/Projects/tensorflow_work/solid_images2/b*.conf')+ glob('/home/zdenek/Projects/tensorflow_work/solid_images/a*.conf')+ glob('/home/zdenek/Projects/tensorflow_work/solid_images2/a*.conf')

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

        for i in range(20000):
            acc,l,_=session.run([fl.accuracy,fl.loss,fl.train],feed_dict={fl.rate: 1e-3})
            print(i,acc,l)

            if i%100 is 0:
                a,true,softmax,acc,l,_=session.run([next_element,fl.outputs,fl.softmax,fl.accuracy,fl.loss,fl.train],
                        feed_dict={fl.rate: 1e-2})
                name='figures/g_%04d.png'%count
                save_softmax(name,true,softmax)
                name='figures/f_%04d.png'%count
                save_images(name,a['images'],a['labels'])
                count+=1

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
