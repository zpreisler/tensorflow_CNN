def __parse__(dataset):
    import tensorflow as tf
    image_string=tf.read_file(dataset['inputs'])
    image_decoded=tf.image.decode_png(image_string,3)
    image_resized=tf.image.resize_images(image_decoded,[256,256])
    image_cropped=tf.image.central_crop(image_resized,0.5)

    images=image_cropped/tf.reduce_max(image_cropped)

    images=tf.image.random_flip_left_right(images)
    images=tf.image.random_flip_up_down(images)
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

    train_dataset=dataset.repeat().shuffle(length).batch(batch)

    iterator=tf.data.Iterator.from_structure(
            train_dataset.output_types,
            train_dataset.output_shapes)
    next_element=iterator.get_next()
    init_train_op=iterator.make_initializer(train_dataset)

    return next_element,init_train_op
