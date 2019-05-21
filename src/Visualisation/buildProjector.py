from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as plt
import numpy as np
import os
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
from matplotlib.figure import Figure

#creer un tenseur d'images de profils journaliers de consommation à la granularité 30 minutes (1/2 heure)
def createLoadProfileImages(x,x_hat,nPoints):
    images=[]
    xAxis=np.arange(0, 24, 0.5)
    #for index in range(0,calendar_info.shape[0]):
    for index in range(0,nPoints):
    #for index in range(0,5):
        #fig, ax = plt.subplots(dpi=30,figsize=(3,3))
        #fig, ax = plt.subplots(figsize=(3,3))
        fig, ax = plt.subplots(1, 1,figsize=(6,6),dpi=30)

        #ax.set_aspect('equal')
        ax.set_facecolor('whitesmoke')#'xkcd:salmon'
        ax.plot(xAxis,x[index,],'-r')
        ax.plot(xAxis,x_hat[index,],'-b')
        canvas = FigureCanvas(fig)
        canvas.draw()       # draw the canvas, cache the renderer
        width, height = fig.get_size_inches() * fig.get_dpi() #and img = np.fromstring(canvas.to_string_rgb(), dtype='uint8').reshape(height, width, 3)
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height),int(width), 3)
        images.append(image)
        #plt.savefig(os.path.join(log_dir, str(index)+'_fig.png'))
        plt.close(fig)
        plt.gcf().clear()
        #canvas.clear()
    images = np.array(images)
    return images

#converti un tenseur d'images au format "sprite" utilisé par tensorboard
def images_to_sprite(data):
        """Creates the sprite image along with any necessary padding

        Args:
          data: NxHxW[x3] tensor containing the images.

        Returns:
          data: Properly shaped HxWx3 image with any necessary padding.
        """
        if len(data.shape) == 3:
            data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
        data = data.astype(np.float32)
        min = np.min(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
        max = np.max(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
        # Inverting the colors seems to look better for MNIST
        # data = 1 - data

        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, 0),
                   (0, 0)) + ((0, 0),) * (data.ndim - 3)
        # padding = ((0, 0), (0, 0),
        #        (0, 0)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                      constant_values=0)
        # Tile the individual thumbnails into an image.
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                               + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        data = (data * 255).astype(np.uint8)
        return data

#creer un fichier de metadata des features que l'on souhaite visualiser et explorer au sein de la visualisation de la projection de tensorboard
def writeMetaData(log_dir,x_conso,calendar_info,nPoints,has_Odd=False,has_nonWorkingDays=False):
    metadata_path = os.path.join(log_dir, 'df_labels.tsv')

	
    with open(metadata_path, 'w') as metadata_file:
        metadata_file.write('"Date"\t"MaxTemperature\t"MinTemperature\t"Month"\t"WeekDay"\t"is_WeekDay"\t"Holiday"\t"Index"\t"Snows"\t"Floods"\t"Storms"\t"Hurricanes"\t"Rains\t"Colder"\t"Hotter"\t"OddWeekday"\t"OddHoliday"\t"OddTemp"\t"OddNeighbor"\t"HD_predicted"\t"nonWorkingDay"\t"ToTag"\n')
        for index in range(0,nPoints):
            #print(index)
            is_hd=calendar_info.loc[index,'is_holiday_day']
            date=calendar_info.loc[index,'ds']#.str
            if is_hd:
                label="Holiday"
            else:
                label="Day"
            temperatureMax=max(x_conso.loc[index*48:(index+1)*48-1,'temperature_France'])
            temperatureMin=min(x_conso.loc[index*48:(index+1)*48-1,'temperature_France'])
            weekday=calendar_info.loc[index,'weekday']
            month=calendar_info.loc[index,'month']
            isWeekday=calendar_info.loc[index,'is_weekday']
            
            Snows = calendar_info.loc[index,'snow']
            Floods = calendar_info.loc[index,'floods']
            Storms = calendar_info.loc[index, 'storm']
            Hurricanes = calendar_info.loc[index, 'hurricane']
            Rains = calendar_info.loc[index, 'rain']
            Colds = calendar_info.loc[index, 'cold']
            Hots = calendar_info.loc[index, 'hot']
	    
            isOddWeekday=0
            isOddHoliday=0
            isOddTemp=0
            isOddNeighbor=0
            isHDPredicted=0
            isnonWorkingDay=0
            ToTag=0
        
            if(has_Odd):
                isOddWeekday=calendar_info.loc[index,'oddWeekDays']
                isOddHoliday=calendar_info.loc[index,'oddHolidays']
                isOddTemp=calendar_info.loc[index,'oddTemp']
                isOddNeighbor=calendar_info.loc[index,'oddNeighbor']
                isHDPredicted=calendar_info.loc[index,'HD_predicted']
            if(has_nonWorkingDays):
                isnonWorkingDay=calendar_info.loc[index,'nonWorkingDay']
            #label = calendar_info.loc[index,'is_hd']
            #metadata_file.write('{}\t{}\n'.format(index+1, label))
            
            metadata_file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(date,temperatureMax,temperatureMin, month,weekday,isWeekday, label,index+1, Snows, Floods, Storms, Hurricanes, Rains, Colds, Hots,isOddWeekday,isOddHoliday,isOddTemp,isOddNeighbor,isHDPredicted,isnonWorkingDay,ToTag))



#creer un projecteur de l'espace latent x de l'autoencoder
def buildProjector(x,images,log_dir,tensor_name=None):
## Running TensorFlow Session
    tf_data = tf.Variable(x)
    #with tf.InteractiveSession() as sess:
    sess = tf.InteractiveSession()
    #with tf.Session() as sess:
    saver = tf.train.Saver([tf_data])
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    sess.run(tf_data.initializer)
     
    file_name='tf_data.ckpt'
    if(tensor_name):
        file_name=tensor_name+'_tf_data.ckpt'
    saver.save(sess, os.path.join(log_dir, file_name))
    config = projector.ProjectorConfig()

    #e can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = tf_data.name

     # Link this tensor to its metadata(Labels) file
     #embedding.metadata_path = metadata
      # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(log_dir, 'df_labels.tsv')
     # Comment out if you don't want sprites
    if(images is not None):
        embedding.sprite.image_path = os.path.join(log_dir, 'sprite_4_classes.png')
        embedding.sprite.single_image_dim.extend([int(images.shape[1]), int(images.shape[2])])

     # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)
