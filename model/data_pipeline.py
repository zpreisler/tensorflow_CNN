def __parse__(dataset):
    import tensorflow as tf
    from numpy import random
    image_string=tf.read_file(dataset['inputs'])
    image_decoded=tf.image.decode_png(image_string,1)

    image_cropped=tf.image.central_crop(image_decoded,0.5)
    image_resized=tf.image.resize_images(image_cropped,[128,128])

    images=image_resized/tf.reduce_max(image_resized)
    images=tf.image.random_flip_up_down(images)
    images=tf.image.random_flip_left_right(images)
    images=tf.image.rot90(images)

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
    print(files,labels)

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

def get_grid(n=2):
    d=1.0/n
    b=[]
    for i in range(n):
        ii=i+1
        for j in range(n):
            jj=j+1
            b+=[[i*d,j*d,ii*d,jj*d]]
    return b

def __parse2__(dataset):
    import tensorflow as tf
    from numpy import random
    image_string=tf.read_file(dataset['inputs'])
    image_decoded=tf.image.decode_png(image_string,1)

    image=tf.expand_dims(image_decoded,0)

    boxes=get_grid(n=1)
    boxes+=get_grid(n=12)
    box_ind=([0]*len(boxes))
    print(boxes)
    size=[128,128]
    images=tf.image.crop_and_resize(image,boxes,box_ind,size)

    images/=tf.reduce_max(images)

    labels=[tf.one_hot(dataset['outputs'],10)]
    labels=tf.stack(labels*(len(boxes)+1))
    print(images)
    print(labels)

    return {'images':images,'labels':labels}

def data_eval_pipeline(files=None):
    import tensorflow as tf
    from numpy import linspace,zeros,array
    """
    eval dataset
    """
    files,labels=get_labels_from_names(files)
    print(files,labels)

    dataset=tf.data.Dataset.from_tensor_slices( 
            {'inputs': files,
                'outputs': labels}
            )
    dataset=dataset.map(__parse2__)

    eval_dataset=dataset

    iterator=tf.data.Iterator.from_structure(
            eval_dataset.output_types,
            eval_dataset.output_shapes)
    next_element=iterator.get_next()
    init_train_op=iterator.make_initializer(eval_dataset)

    return next_element,init_train_op
