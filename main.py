#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def main(argv):
    print("Convolutional Neural Network")
    import tensorflow as tf
    from model.data_pipeline import data_pipeline,get_labels,get_labels_from_names
    from model.data_plot import plot_images,save_images,plot_softmax,save_softmax,plot_images_softmax,save_images_softmax
    from model.model import discriminator
    from glob import glob

    #files=(glob('/home/zdenek/Projects/tensorflow_work/solid_images2/b*.conf')
    #        +glob('/home/zdenek/Projects/tensorflow_work/solid_images/a*.conf')
    #        +glob('/home/zdenek/Projects/tensorflow_work/solid_images2/a*.conf')
    #        )

    #files=(glob('gen_images/*.conf'))
    #print(get_labels(files))

    #files=(glob('gen_images/*.png'))
    #print(get_labels_from_names(files))

    files=(glob('gen_images/*.png'))

    next_element,init_op=data_pipeline(files,batch=128)

    print(len(files))


    #inputs=tf.placeholder(tf.float64,[None,96,96,3])
    #outputs=tf.placeholder(tf.float64,[10])

    fl=discriminator(inputs=next_element['images'],outputs=next_element['labels']) 

    save=tf.train.Saver()

    with tf.Session() as session:
        print("Start Session")
        """Init"""

        try:
            save.restore(session,'log/last.ckpt')
        except tf.errors.NotFoundError:
            tf.global_variables_initializer().run(session=session)
            pass

        session.run(init_op)

        from matplotlib.pyplot import figure,show,plot
        count=0

        for i in range(10000):
            acc,l,_=session.run([fl.accuracy,
                fl.loss,fl.train],feed_dict={fl.rate: 1e-4})
            print(i,acc,l)

            if i%100 is 0:

                a,true,softmax,acc=session.run([next_element,
                    fl.outputs,fl.softmax,
                    fl.accuracy])
                
                name='figures/a_%04d.png'%count
                save_images_softmax(name,a['images'],a['labels'],true,softmax)

                name='figures/a_%04d.pdf'%count
                save_images_softmax(name,a['images'],a['labels'],true,softmax)

                count+=1

            if i%200 is 0:
                save.save(session,'log/last.ckpt')

        for i in range(100):
                a,true,softmax,acc=session.run([next_element,
                    fl.outputs,fl.softmax,
                    fl.accuracy])
                
                print(i,acc)

                #plot_images_softmax(a['images'],a['labels'],true,softmax)
                
                name='figures/a_%04d.png'%count
                save_images_softmax(name,a['images'],a['labels'],true,softmax)

                name='figures/a_%04d.pdf'%count
                save_images_softmax(name,a['images'],a['labels'],true,softmax)

                count+=1

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
