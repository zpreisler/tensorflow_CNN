def __parse__(dataset):
    import tensorflow as tf
    from numpy import random
    image_string=tf.read_file(dataset['inputs'])
    image_decoded=tf.image.decode_png(image_string,3)

    image_cropped=tf.image.central_crop(image_decoded,0.5)
    image_resized=tf.image.resize_images(image_cropped,[128,128])

    images=image_cropped/tf.reduce_max(image_cropped)

    #images=image_resized/tf.reduce_max(image_resized)*tf.random_uniform([1],minval=0.9,maxval=1.0)
    #images=image_resized/tf.reduce_max(image_resized)

    #noise=tf.truncated_normal(shape=tf.shape(images),mean=0.0,stddev=0.33,dtype=tf.float32)
    #images=tf.add(images,noise)

    #images=images/tf.reduce_max(images)
    #images=image_cropped/tf.reduce_max(image_cropped)*random.uniform(0.9,1.0)
    #images=tf.image.random_flip_left_right(images)
    #images=tf.image.random_flip_up_down(images)
    #images=tf.image.rot90(images)

    labels=tf.one_hot(dataset['outputs'],10)

    return {'images':images,'labels':labels}

def get_labels(files,ext='.png',key='number_of_patches'):
    from myutils import configuration
    from numpy import array
    c=configuration(files)

    files=[]
    labels=[]

    for d in c.dconf:
        name=''.join(d['path']+d['name']+[ext])
        label=d[key]
        files+=[name]
        labels+=label

    files=array(files)
    labels=array(labels,dtype='int32')

    return files,labels 

def get_labels_from_names(files):
    from numpy import array
    labels=[]

    for f in files:
        label=0
        if f[f.rfind('/')+1] is 'h':
            label=1
        if f[f.rfind('/')+1] is 's':
            label=2
        if f[f.rfind('/')+1] is 'x':
            label=3
        labels+=[label]

    files=array(files)
    labels=array(labels,dtype='int32')

    return files,labels
        

def data_pipeline(files=None,batch=32):
    import tensorflow as tf
    from numpy import linspace,zeros,array
    """
    train dataset
    """

    #files,labels=get_labels(files)
    files,labels=get_labels_from_names(files)

    length=len(files)
    if batch>length:
        batch=length

    dataset=tf.data.Dataset.from_tensor_slices( 
            {'inputs': files,
                'outputs': labels}
            )
    dataset=dataset.map(__parse__)

    train_dataset=dataset.repeat().shuffle(batch).batch(batch)

    iterator=tf.data.Iterator.from_structure(
            train_dataset.output_types,
            train_dataset.output_shapes)
    next_element=iterator.get_next()
    init_train_op=iterator.make_initializer(train_dataset)

    return next_element,init_train_op
