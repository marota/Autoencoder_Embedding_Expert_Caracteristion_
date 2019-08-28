
#nbsample=20
import tensorflow as tf
from keras.callbacks import Callback
from tensorflow.contrib.tensorboard.plugins import projector
from keras import backend as K
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np
from FeaturesScore.scoring import *


class NEpochLogger(Callback):
    def __init__(self,x_train_data, display,x_conso=None,calendar_info=None,is_VAE=True):
        self.seen = 0
        self.display = display
        self.x_train_data = x_train_data
        self.x_conso=x_conso
        self.calendar_info=calendar_info
        self.is_VAE=is_VAE


    def on_epoch_end(self, epoch, logs={}):
        self.seen += logs.get('size', 0)
        #print([l.name for l in self.model.layers])
        
        if epoch % self.display == 0:
            metrics_log = ''
            for k in self.params['metrics']:
                if k in logs:
                    val = logs[k]
                    if abs(val) > 1e-3:
                        metrics_log += ' - %s: %.4f' % (k, val)
                    else:
                        metrics_log += ' - %s: %.4e' % (k, val)
            #weight = self.model.loss.keywords['weight']
            
            if(self.is_VAE):
                weight = K.get_value(self.model.loss_weights['decoder_for_kl'])
            #print(self.model.get_layer('sample_z').ouput.values())

            inputTensor=[self.model.get_layer('x_true').input]
            #inputTensor=[self.model.get_layer('enc_x_true').input,self.model.get_layer('enc_cond').input]
            
            n_cond_pre=0
            n_embs_input=0
            for l in self.model.layers:
                if('cond_pre' in l.name ):
                    inputTensor.append(self.model.get_layer(l.name).input)
                    n_cond_pre=n_cond_pre+1
            
            for l in self.model.layers:
                if('emb_input' in l.name ):
                    inputTensor.append(self.model.get_layer(l.name).input)
                    n_embs_input=n_embs_input+1
                    print(l.name )

            x_inputs =self.x_train_data
            inputsY=x_inputs[0]
            
            if(n_cond_pre>=1):
                if(n_cond_pre==1):
                    cond_pre=x_inputs[1]
                else:
                    cond_pre=x_inputs[1:(1+n_cond_pre)]
                
                if(n_embs_input>=1):
                    emb_inputs=x_inputs[(1+n_cond_pre):]
                    embModel=self.model.get_layer('embedding_enc')
                    emb_ouputs = embModel.predict(emb_inputs)
                
                    cond = np.concatenate((cond_pre, emb_ouputs), axis=1)
                    input_encoder = [inputsY,cond]
                else:
                    input_encoder = [inputsY,cond_pre]
            elif(n_embs_input>=1):
                emb_inputs=x_inputs[1:]
                embModel=self.model.get_layer('embedding_enc')
                emb_ouputs = embModel.predict(emb_inputs)
                #emb_ouputs=np.squeeze(emb_ouputs, axis=0)
                input_encoder = [inputsY,emb_ouputs]
            else:
                input_encoder=[inputsY]
            
            self.response_model=self.model.get_layer('encoder')
            
            responses=self.response_model.predict(input_encoder)
            if(self.is_VAE):
                responses=responses[0]
           
            #responses=self.model.encoder.predict(self.x_train_data)
            print(np.sum(np.abs(responses),axis=0))
            predictFeaturesInLatentSPace(self.x_conso,self.calendar_info,responses,k=5)
            
            valLoss=logs.get('val_loss')
            
            if(self.is_VAE):
                print('{} Epochs ... {} val_loss {} ... lambda Loss {}'.format(epoch, metrics_log,valLoss,weight))
            else:
                print('{} Epochs ... {}'.format(epoch, metrics_log))
            #print('{} Epochs ... {}'.format(epoch, metrics_log))

class callbackWeightLoss(Callback): #to adapt the weights of the loss components
    # customize your behavior
    def __init__(self,beta=0.0,rate=0.002,minimum=0.001):
        self.beta = beta
        self.rate = rate
        self.minimum=minimum
        
    def on_epoch_end(self, epoch, logs={}):
        if(epoch==0 and not self.beta==0.0):
            K.set_value(self.model.loss_weights['decoder_for_kl'],self.beta)
        weightVar=self.model.loss_weights['decoder_for_kl']
        weight=K.get_value(weightVar)
        new_Weight=weight-self.rate*weight#0.99*np.cos(epoch/360*2*Pi)
        #if(new_Weight>=10000*self.beta ):
        #    new_Weight=100*self.beta 
        if(new_Weight<=self.minimum):
            new_Weight=self.minimum
        K.set_value(weightVar,new_Weight)


class TensorResponseBoard(TensorBoard):
    def __init__(self, nPoints, img_path, img_size, **kwargs):
        #super(TensorResponseBoard, self).__init__(**kwargs)
        super().__init__(**kwargs)
        #self.val_size = val_size
        self.img_path = img_path
        self.img_size = img_size
        self.nPoints=nPoints

    def set_model(self, model):
        super().set_model(model)
        #super(TensorResponseBoard, self).set_model(model)

        if self.embeddings_freq and self.embeddings_layer_names:
            embeddings = {}
            print([l.name for l in model.layers])
            lays_dec=self.model.get_layer('decoder')
            print([l.name for l in lays_dec.layers])
            
            layer_name=self.embeddings_layer_names[0]
            
            # initialize tensors which will later be used in `on_epoch_end()` to
            # store the response values by feeding the val data through the model
                
            #we suppose that we look for a layer in the decoder
            layer = self.model.get_layer('decoder').get_layer(layer_name)
                
            output_dim = layer.output.shape[-1]
            response_tensor = tf.Variable(tf.zeros([self.nPoints, output_dim]),
                                              name=layer_name + '_response')
            embeddings[layer_name] = response_tensor

            self.embeddings = embeddings
            
            #self.saver = tf.train.Saver(list(self.embeddings.values()))
            self.saver = tf.train.Saver(list(self.embeddings.values()))#tf.train.Saver([tf_data])
            
            response_outputs = [self.model.get_layer('decoder').get_layer(layer_name).output
                                for layer_name in self.embeddings_layer_names]
            
            #self.response_model=tf.Variable(x)
            response_inputs=[self.model.get_layer('x_true').input,self.model.get_layer('cond_pre').input]
            for l in model.layers:
                if('emb_input' in l.name ):
                    response_inputs.append(self.model.get_layer(l.name).input)
            #['emb_input_0', 'emb_input_1', 'cond_pre', 'embedding', 'x_true', 'conc_cond', 'encoder', 'sample_z', 'decoder', 'decoder_for_kl']
            #print(self.model.inputs)
            #self.response_model = Model(self.model.inputs, response_outputs)
            
            #self.response_model=Model(response_inputs,response_outputs)
            
            config = projector.ProjectorConfig()
            embeddings_metadata = {layer_name: self.embeddings_metadata
                                   for layer_name in embeddings.keys()}

            for layer_name, response_tensor in self.embeddings.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = response_tensor.name

                # for coloring points by labels
                embedding.metadata_path = embeddings_metadata[layer_name]

                # for attaching images to the points
                #embedding.sprite.image_path = self.img_path
                #embedding.sprite.single_image_dim.extend(self.img_size)

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        super(TensorResponseBoard, self).on_epoch_end(epoch, logs)
        #super().on_epoch_end(epoch, logs)
        print(self.xy.x_train)
        print(self.embeddings.values())
        
        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                # feeding the validation data through the model
                val_data = self.xy.x_train#self.validation_data[0]
                print(self.xy.x_train)
                print(self.embeddings.values())
                #_encoded = model2.encoder.predict(input_encoder)[0]
                
                #response_values = self.model.get_layer('decoder').get_layer(layer_name).output.get_value()#self.response_model.predict(val_data)
                response_values=self.embeddings.values()
                
                # record the response at each layers we're monitoring
                response_tensors = []
                for layer_name in self.embeddings_layer_names:
                    response_tensors.append(self.embeddings[layer_name])
                K.batch_set_value(list(zip(response_tensors, response_values)))

                # finally, save all tensors holding the layer responses
                self.saver.save(self.sess, self.embeddings_ckpt_path, epoch)


def p_d_kernel(a,b,h=1):
    return K.exp(-0.5 * K.square((a - b)/h)) #Gaussian
    #return K.exp(-(K.abs(a - b)) / h) #Laplacian
    #return K.sum(a * b, ax is = -1) #Linear

def Gram_matrix(x,h):
    N = len(x)
    Dist_M = np.exp(-0.5 * np.square((x.reshape(-1,1)-x.reshape(1,-1))/h))
    Norm = np.sqrt(np.diag(Dist_M).reshape(-1,1) * np.diag(Dist_M).reshape(1,-1))
    return Dist_M/(N*Norm)

def join_Gram_Matrix(list_M):
    N = list_M[0].shape[1]
    AB = np.ones(shape = list_M[0].shape)
    for i in range(len(list_M)):
        AB = AB * list_M[i]
    trace_AB = np.trace(AB)
    return AB / trace_AB

def Renyi_entropy(A, alpha):
    A_eigval =  np.linalg.eigvalsh(A)
    return np.log(np.sum(np.nan_to_num(A_eigval**alpha))) / np.log(2) / (1 - alpha)

def Silverman_rule(h,n,d):
    return h * (n**(-1/(4+d)))
                               
class InformationHistory(Callback):
    def __init__(self,h,alpha, dataset_train, emb=False):
        self.h = h
        self.alpha = alpha
        self.dataset_train = dataset_train
        self.emb=emb

        self.reconstruction_MI = []
        self.latent_entropy = []
        self.latent_MI = []
        if self.dataset_train[1].shape[-1] > 1:
            self.cond_MI = []

        x = self.dataset_train[0]
        cond_input = self.dataset_train[1]

        self.Gram_x = join_Gram_Matrix([Gram_matrix(x[:,j], Silverman_rule(self.h, x.shape[0], x.shape[1])) for j in range(x.shape[1])])

        if len(self.dataset_train)>2:
            all_conds = np.concatenate([self.dataset_train[i] for i in range(1,len(self.dataset_train))], axis=1)
            self.Gram_cond = join_Gram_Matrix([Gram_matrix(all_conds[:,j], Silverman_rule(self.h, x.shape[0], x.shape[1])) for j in range(all_conds.shape[1])])
        else:
            self.Gram_cond = join_Gram_Matrix([Gram_matrix(cond_input[:,j], Silverman_rule(self.h, x.shape[0], x.shape[1])) for j in range(cond_input.shape[1])])

    def on_epoch_end(self, epoch, logs={}):
        lays_enc = self.model.get_layer('encoder')
        x = self.dataset_train[0]

        if self.emb == False:
            input_encoder = self.dataset_train
        else:
            lays_emb = self.model.get_layer('embedding_enc')
            emb_inputs = lays_emb.predict([self.dataset_train[i] for i in range(1,len(self.dataset_train))])
            input_encoder = [x,emb_inputs]

        x_encoded = lays_enc.predict(input_encoder)[0]
        x_hat = self.model.predict(self.dataset_train)[0]

        Gram_x_encoded = join_Gram_Matrix([Gram_matrix(x_encoded[:,j], Silverman_rule(self.h, x.shape[0], x.shape[1])) for j in range(x_encoded.shape[1])])
        Gram_x_hat = join_Gram_Matrix([Gram_matrix(x_hat[:,j], Silverman_rule(self.h, x.shape[0], x.shape[1])) for j in range(x_hat.shape[1])])

        self.latent_entropy.append(Renyi_entropy(Gram_x_encoded, self.alpha))
        self.reconstruction_MI.append(Renyi_entropy(Gram_x_hat, self.alpha) + Renyi_entropy(self.Gram_x, self.alpha) - Renyi_entropy(join_Gram_Matrix([Gram_x_hat, self.Gram_x]), self.alpha))
        self.cond_MI.append(self.latent_entropy[-1] + Renyi_entropy(self.Gram_cond, self.alpha) - Renyi_entropy(join_Gram_Matrix([Gram_x_encoded, self.Gram_cond]), self.alpha))
        self.latent_MI.append(self.latent_entropy[-1] + Renyi_entropy(self.Gram_x, self.alpha) - Renyi_entropy(join_Gram_Matrix([Gram_x_encoded, self.Gram_x]), self.alpha))


        if epoch % 100 ==0:
            print('epoch {} : latent {}, reconstruction {}'.format(epoch,self.latent_entropy[-1], self.reconstruction_MI[-1]))


#    tf_data = tf.Variable(x)
#    with tf.Session() as sess:
#        saver = tf.train.Saver([tf_data])
#        sess.run(tf_data.initializer)
#        
#        file_name='tf_data.ckpt'
#        if(tensor_name):
#            file_name=tensor_name+'_tf_data.ckpt'
#        saver.save(sess, os.path.join(log_dir, file_name))
#        config = projector.ProjectorConfig()
#
#    # One can add multiple embeddings.
#        embedding = config.embeddings.add()
#        embedding.tensor_name = tf_data.name
#
#        # Link this tensor to its metadata(Labels) file
#        #embedding.metadata_path = metadata
#         # Link this tensor to its metadata file (e.g. labels).
#        embedding.metadata_path = os.path.join(log_dir, 'df_labels.tsv')
#        # Comment out if you don't want sprites
#        if(images):
#            embedding.sprite.image_path = os.path.join(log_dir, 'sprite_4_classes.png')
#            embedding.sprite.single_image_dim.extend([int(images.shape[1]), int(images.shape[2])])
#
#        # Saves a config file that TensorBoard will read during startup.
#        projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)
                

                
