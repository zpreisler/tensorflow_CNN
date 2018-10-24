#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def main(argv):
    print("Convolutional Neural Network")
    from numpy import array,stack
    from pprint import pprint
    from tensorflow_utils import image_pipeline,get_labels_from_filenames,discriminator
    from matplotlib.pyplot import imshow,figure,show
    from glob import glob

    steps=1000
    batch_size=128
    rate=1e-4

    train_files=glob("images/scale=8.5/rotate/*.png")
    train_labels=get_labels_from_filenames(train_files)

    eval_files=glob("images/scale=8.5/eval/*.png")
    eval_labels=get_labels_from_filenames(eval_files)

    handle,image,train_image_op,eval_image_op=image_pipeline(
            {'images': train_files,'labels': train_labels},
            {'images': eval_files,'labels': eval_labels},
            batch_size=batch_size)

    d=discriminator(image['images'],k=5,l=5)
    d.define_output(image['labels'])

    save=tf.train.Saver()

    with tf.Session() as session:
        print("Run")
        try:
            save.restore(session,'log/last.ckpt')
        except tf.errors.NotFoundError:
            tf.global_variables_initializer().run(session=session)
            pass

        training_handle=session.run(train_image_op.string_handle())
        evaluation_handle=session.run(eval_image_op.string_handle())

        session.run(train_image_op.initializer)
        session.run(eval_image_op.initializer)

        for step in range(steps):
            loss,accuracy,_=session.run([d.loss,
                d.accuracy,
                d.train],
                feed_dict={ d.rate: rate,
                    handle: training_handle})
            print("[{}] {:2.4f}: {:1.4f}".format(step,loss,accuracy))

            if step%5 is 0:
                loss,accuracy,true,prediction=session.run([d.loss,d.accuracy,d.true,d.prediction],
                    feed_dict={handle: evaluation_handle})
                print("\t\t\t{:2.4f}: {:2.4}\n {} {}".format(loss,accuracy,true,prediction))

            if step%5 is 0:
                save.save(session,'log/last.ckpt')

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
