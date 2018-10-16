def plot_images(image,label,x=5,y=5):
    from matplotlib.pyplot import imshow,figure,show,subplots,subplots_adjust
    fig,axes=subplots(x,y,figsize=(12,12))
    count=0
    for n in range(x):
        for m in range(y):
            axes[n,m].imshow(image[count])
            axes[n,m].set_title("%d"%label[count].argmax(),color='g')
            count+=1

    subplots_adjust(left=0.03,
            right=0.97,
            bottom=0.03,
            top=0.97,
            wspace=0.22,
            hspace=0.22)
    show()

def save_images(name,image,label,x=5,y=5):
    from matplotlib.pyplot import imshow,figure,show,subplots,subplots_adjust,savefig,close
    fig,axes=subplots(x,y,figsize=(12,12))
    count=0
    for n in range(x):
        for m in range(y):
            axes[n,m].imshow(image[count])
            axes[n,m].set_title("%d"%label[count].argmax(),color='g')
            count+=1

    subplots_adjust(left=0.03,
            right=0.97,
            bottom=0.03,
            top=0.97,
            wspace=0.22,
            hspace=0.22)
    savefig(name)
    close()

def plot_softmax(true,softmax,x=5,y=5):
    from matplotlib.pyplot import imshow,figure,show,subplots,subplots_adjust
    fig,axes=subplots(x,y,figsize=(12,12))
    count=0
    for n in range(x):
        for m in range(y):
            axes[n,m].plot(true[count])
            axes[n,m].plot(softmax[count])
            count+=1

    subplots_adjust(left=0.03,
            right=0.97,
            bottom=0.03,
            top=0.97,
            wspace=0.22,
            hspace=0.22)
    show()

def save_softmax(name,true,softmax,x=5,y=5):
    from matplotlib.pyplot import imshow,figure,show,subplots,subplots_adjust,savefig,close
    fig,axes=subplots(x,y,figsize=(12,12))
    count=0
    for n in range(x):
        for m in range(y):
            axes[n,m].plot(true[count])
            axes[n,m].plot(softmax[count])
            count+=1

    subplots_adjust(left=0.03,
            right=0.97,
            bottom=0.03,
            top=0.97,
            wspace=0.22,
            hspace=0.22)

    savefig(name)
    close()
