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

def plot_images_softmax(image,label,true,softmax,x=5,y=5):
    from matplotlib.pyplot import imshow,figure,show,subplots,subplots_adjust,bar
    from numpy import arange
    fig,axes=subplots(x,y,figsize=(12,12))
    count=0
    for n in range(x):
        for m in range(y):
            ax1=axes[n,m]
            ax1.imshow(image[count],origin='lower')

            ax1.set_xlim(0,63)
            ax1.set_ylim(0,63)

            ax2=axes[n,m].twiny()
            x=arange(len(softmax[count]))+0.5
            ax2.bar(x,true[count]*47,width=0.9,edgecolor='k',linewidth=0.9)
            ax2.bar(x,softmax[count]*47,width=0.9,edgecolor='k',linewidth=0.9)

            for a,b in zip(x,softmax[count]):
                if b>0.35: 
                    ax2.text(a+0.09,1,r"$[%d]\,\, %.2lf$"%(a,b),ha='center',va='bottom',rotation=90)
                elif b>0.15:
                    ax2.text(a+0.09,1,r"$[%d]$"%(a),ha='center',va='bottom',rotation=90)

            ax2.set_xlim(0,10)

            axes[n,m].set_title("%d"%label[count].argmax(),color='g')
            count+=1

    subplots_adjust(left=0.03,
            right=0.97,
            bottom=0.03,
            top=0.97,
            wspace=0.22,
            hspace=0.22)
    show()

def save_images_softmax(name,image,label,true,softmax,x=5,y=5):
    from matplotlib.pyplot import imshow,figure,show,subplots,subplots_adjust,bar,savefig,close
    from numpy import arange
    fig,axes=subplots(x,y,figsize=(12,12))
    count=0
    for n in range(x):
        for m in range(y):
            ax1=axes[n,m]
            ax1.imshow(image[count],origin='lower')

            ax1.set_xlim(0,63)
            ax1.set_ylim(0,63)

            ax2=axes[n,m].twiny()
            x=arange(len(softmax[count]))+0.5
            ax2.bar(x,true[count]*47,width=0.9,edgecolor='k',linewidth=0.9)
            ax2.bar(x,softmax[count]*47,width=0.9,edgecolor='k',linewidth=0.9)

            for a,b in zip(x,softmax[count]):
                if b>0.35: 
                    ax2.text(a+0.09,1,r"$[%d]\,\, %.2lf$"%(a,b),ha='center',va='bottom',rotation=90)
                elif b>0.15:
                    ax2.text(a+0.09,1,r"$[%d]$"%(a),ha='center',va='bottom',rotation=90)

            ax2.set_xlim(0,10)

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
