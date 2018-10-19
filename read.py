#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def get_grid(n=2):
    d=1.0/n
    b=[]
    for i in range(n):
        ii=i+1
        for j in range(n):
            jj=j+1
            b+=[[i*d,j*d,ii*d,jj*d]]
    return b

def main(argv):
    print("Convolutional Neural Network")
    import tensorflow as tf
    from model.data_pipeline import data_pipeline,get_labels,get_labels_from_names
    from model.data_plot import plot_images,save_images,plot_softmax,save_softmax,plot_images_softmax,save_images_softmax
    from model.model import discriminator
    from glob import glob

    files=(glob('gen_images/evaluate/hc20_83.png'))

    dataset=tf.data.Dataset.from_tensor_slices(files)

    with tf.Session() as session:
        print(files)
        image_string=tf.read_file(files[0])
        image_decoded=tf.image.decode_png(image_string,1)

        image=[]
        for i in range(3):
            image_cropped=tf.image.central_crop(image_decoded,0.5)

            image_cropped=image_cropped/tf.reduce_max(image_cropped)

            image+=[image_cropped]
        image_resized=tf.image.resize_images(image_cropped,[128,128])

        print(image_resized)
        t=tf.stack(image)

        image_decoded=image_decoded/tf.reduce_max(image_decoded)
        t=tf.expand_dims(image_decoded,0)
        print(t)
        
        b=tf.constant([[[0.3,0.3,0.6,0.6]]],dtype=tf.float32)
        b2=tf.constant([[[0.1,0.1,0.5,0.5]]],dtype=tf.float32)
        boxes=tf.concat([b,b2],1)
        #boxes=tf.concat([boxes,boxes,boxes],0,name="boxes")
        print(boxes)

        tt=tf.image.draw_bounding_boxes(t,boxes)

        #b=tf.constant([[0.3,0.3,0.6,0.6],[0.1,0.1,0.5,0.5]])
        #d=tf.constant([128,128])
        #b=[[0.3,0.3,0.6,0.6],[0.1,0.1,0.5,0.5]]

        n=4
        b=get_grid(n=n)

        c=([0]*len(b))
        d=[128,128]
        print(b)
        print(c)
        print(d)

        ttt=tf.image.crop_and_resize(t,b,c,d)

        
        tt=tf.image.grayscale_to_rgb(tt)
        ttt=tf.image.grayscale_to_rgb(ttt)

        a=session.run(tt)
        b=session.run(ttt)

        print(b.shape)

        from matplotlib.pyplot import imshow,show,figure
        figure()
        imshow(a[0])

        plot_images(b,x=n,y=n,
                wspace=.0,hspace=.0)
        
        show()

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
