import os
import json
from matplotlib import pyplot as plt
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Dropout, BatchNormalization, Activation
from keras.layers.merge import concatenate, Add, Multiply
from keras import backend as K
from keras.callbacks import Callback
from keras import losses
from keras import optimizers
import tensorflow as tf
from functools import partial, update_wrapper


class BaseModel():
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        if 'name' not in kwargs:
            raise Exception('Please specify model name!')

        self.name = kwargs['name']

        if 'output' not in kwargs:
            self.output = 'output'
        else:
            self.output = kwargs['output']

        self.trainers = {}
        self.history = None

    def save_model(self, out_dir):
        folder = os.path.join(out_dir)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            v.save_weights(filename)

    def store_to_save(self, name):
        self.trainers[name] = getattr(self, name)

    def load_model(self, folder):
        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            getattr(self, k).load_weights(filename)

    def main_train(self, dataset, training_epochs=100, batch_size=100, callbacks=[],validation_data=None, verbose=0,validation_split=None):

        out_dir = os.path.join(self.output, self.name)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        res_out_dir = os.path.join(out_dir, 'results')
        if not os.path.isdir(res_out_dir):
            os.mkdir(res_out_dir)

        wgt_out_dir = os.path.join(out_dir, 'models')
        if not os.path.isdir(wgt_out_dir):
            os.mkdir(wgt_out_dir)

        #if 'test' in dataset.keys():
        #    validation_data = (dataset['test']['x'], dataset['test']['y'])
        #else:
        #    validation_data = None

        print('\n\n--- START TRAINING ---\n')
        history = self.train(dataset['train'], training_epochs, batch_size, callbacks, validation_data=validation_data, verbose=verbose,validation_split=validation_split)

        self.history = history.history
        self.save_model(wgt_out_dir)
        self.plot_loss(res_out_dir)

        with open(os.path.join(res_out_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f)

    def plot_loss(self, path_save = None):

        nb_epoch = len(self.history['loss'])

        if 'val_loss' in self.history.keys():
            best_iter = np.argmin(self.history['val_loss'])
            min_val_loss = self.history['val_loss'][best_iter]

            plt.plot(range(nb_epoch), self.history['val_loss'], label='test (min: {:0.2f}, epch: {:0.2f})'.format(min_val_loss, best_iter))

        plt.plot(range(nb_epoch), self.history['loss'], label = 'train')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('loss evolution')
        plt.legend()

        if path_save is not None:
            plt.savefig(os.path.join(path_save, 'loss_evolution.png'))

    #abstractmethod
    def train(self, training_dataset, training_epochs, batch_size, callbacks, validation_data=None, verbose=0,validation_split=None):
        '''
        Plase override "train" method in the derived model!
        '''

        pass
    

#un modèle CVAE ou l'on passe les conditions mais sans embedding
class Guided_VAE(BaseModel):
    def __init__(self, input_dim=[96], cond_dims=[12], z_dim=2, e_dims=[24], d_dims=[24], beta=1, embeddingBeforeLatent=False,pDropout=0.0, verbose=True,is_L2_Loss=True, InfoVAE = False, gamma=0, prior='Gaussian',has_skip=True,has_BN=1, lr = 0.001,**kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.cond_dims = cond_dims
        self.z_dim = z_dim
        self.e_dims = e_dims
        self.d_dims = d_dims
        self.beta = beta
        self.dropout = pDropout#la couche de dropout est pour l'instant commentée car pas d'utilité dans les experiences
        self.encoder = None
        self.decoder = None
        self.latent=None
        self.cvae = None
        self.embeddingBeforeLatent=embeddingBeforeLatent#in the decoder, do a skip only with the embedding and not the latent space to make it more influencial
        self.verbose = verbose
        self.losses={}
        self.weight_losses={}
        self.is_L2_Loss=is_L2_Loss
        self.has_skip=has_skip
        self.lr=lr
        self.InfoVAE = InfoVAE
        if self.InfoVAE:
            self.gamma= gamma
        self.has_BN=has_BN
        self.prior = prior

        self.build_model()

    def build_model(self):
        """

        :param verbose:
        :return:
        """

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        if self.InfoVAE:
            print('InfoVAE : ', str(self.InfoVAE))

        x_true = Input(shape=(self.input_dim[0],), name='x_true')
        cond_true = [Input(shape=(cond_dim,)) for cond_dim in self.input_dim[1:]]
        x_inputs = [x_true] + cond_true
        # Encoding
        z_mu, z_log_sigma = self.encoder(x_inputs)
        #self.latent=Lambda(lambda x:x,'latent')(z_mu)
        
        x_true_inputs = Input(shape=(self.input_dim[0],), name='x_true_zmu_Layer') 
        x = Lambda(lambda x: x,name='z_mu')(x_true_inputs)
        self.latent=Model(inputs=[x_true_inputs], outputs=[x], name='z_mu_output')
        ZMU=self.latent(z_mu)



        # Sampling
        # Here if the prior is Gaussian, z_log_sigma is the log of sigma², whereas it refers to the log of sigma if it's Laplacian. 
        def sample_z(args):
            mu, log_sigma = args
            if self.prior=='Gaussian':
                eps = K.random_normal(shape=(K.shape(mu)[0], self.z_dim), mean=0., stddev=1.)
                return mu + K.exp(log_sigma / 2) * eps
            elif self.prior=='Laplace':
                U = K.random_uniform(shape=(K.shape(mu)[0], self.z_dim), minval =0.0, maxval=1.)
                V = K.random_uniform(shape=(K.shape(mu)[0], self.z_dim), minval =0.0, maxval=1.)
                Rad_sample = 2.*K.cast(K.greater_equal(V,0.5), dtype='float32') - 1. 
                Expon_sample = -K.exp(log_sigma)*K.log(1-U)
                return mu + Rad_sample*Expon_sample

        z = Lambda(sample_z, name='sample_z')([z_mu, z_log_sigma])

        # Decoding
        x_hat= self.decoder(z)
        
        #identity layer to have two output layers and compute separately 2 losses (the kl and the reconstruction)
             
        x = Lambda(lambda x: x)(x_true_inputs)
        identitModel=Model(inputs=[x_true_inputs], outputs=[x], name='decoder_for_kl')
        if self.InfoVAE :
            identitModel2=Model(inputs=[x_true_inputs], outputs=[x], name='decoder_info')
            xhatTer = identitModel2(x_hat)

        
        xhatBis=identitModel(x_hat)

        # Defining loss
        if self.InfoVAE:
            # Defining loss
            vae_loss, recon_loss, kl_loss, info_loss= self.build_loss_info(z_mu, z_log_sigma, z, beta=self.beta, gamma=self.gamma)

            # Defining and compiling cvae model
            self.losses = {"decoder": recon_loss,"decoder_for_kl": kl_loss, "decoder_info": info_loss}
            #lossWeights = {"decoder": 1.0, "decoder_for_kl": 0.01}
            self.weight_losses = {"decoder": 1.0, "decoder_for_kl": self.beta, "decoder_info":self.gamma}

            Opt_Adam = optimizers.Adam(lr=self.lr)
            
            self.cvae = Model(inputs=x_inputs, outputs=[x_hat,xhatBis, xhatTer])#self.encoder.outputs])
            self.cvae.compile(optimizer=Opt_Adam,loss=self.losses,loss_weights=self.weight_losses)
            
        else:
            vae_loss, recon_loss, kl_loss = self.build_loss(z_mu, z_log_sigma, beta=self.beta)

            # Defining and compiling cvae model
            self.losses = {"decoder": recon_loss,"decoder_for_kl": kl_loss}
            #lossWeights = {"decoder": 1.0, "decoder_for_kl": 0.01}
            self.weight_losses = {"decoder": 1.0, "decoder_for_kl": self.beta}

            Opt_Adam = optimizers.Adam(lr=self.lr)
            
            self.cvae = Model(inputs=x_inputs, outputs=[x_hat,xhatBis])#self.encoder.outputs])
            self.cvae.compile(optimizer=Opt_Adam,loss=self.losses,loss_weights=self.weight_losses)
            
            
        # Store trainers
        self.store_to_save('cvae')
        self.store_to_save('encoder')
        self.store_to_save('decoder')

        if self.verbose:
            print("complete model: ")
            self.cvae.summary()
            print("encoder: ")
            self.encoder.summary()
            print("decoder: ")
            self.decoder.summary()

    def build_encoder(self):
        """
        Encoder: Q(z|X,y)
        :return:
        """
        x_inputs = []
        mu_cond=[]

        for i,input_dim in enumerate(self.input_dim):
            if i==0:
                x = Input(shape=(input_dim,), name='enc_x_true')
                x_inputs.append(x)

                class_cond = Lambda(lambda x: K.one_hot(K.constant(i,dtype='int32',shape=(1,)), len(self.cond_dims)), name="latent_dense_mu_c_class_{}".format(i))(x)

                nLayers = len(self.e_dims)
                for idx, layer_dim in enumerate(self.e_dims):
                    #x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
                    x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
                    #x = Dropout(self.dropout)(x)

                #z_mu = Dense(units=self.z_dim, activation='linear', name="latent_dense_mu")(x)
                #z_log_sigma = Dense(units=self.z_dim, activation='linear', name='latent_dense_log_sigma')(x)
                #x = Dense(units=self.z_dim, activation='relu', name="enc_dense_zdim")(x)
                z_mu = Dense(units=self.z_dim, activation='linear', name="latent_dense_mu")(x)
                z_log_sigma = Dense(units=self.z_dim, activation='linear', name='latent_dense_log_sigma')(x)

            else :
                x = Input(shape=(input_dim,), name='enc_cond_{}'.format(i))
                x_inputs.append(x)

                nLayers = len(self.cond_dims[i-1])
                for idx, layer_dim in enumerate(self.cond_dims[i-1]):
                    #x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
                    x = Dense(units=layer_dim, activation='relu', name="enc_cond_{}_dense_{}".format(i,idx))(x)
                    #x = Dropout(self.dropout)(x)

                #z_mu = Dense(units=self.z_dim, activation='linear', name="latent_dense_mu")(x)
                #z_log_sigma = Dense(units=self.z_dim, activation='linear', name='latent_dense_log_sigma')(x)
                #x = Dense(units=self.z_dim, activation='relu', name="enc_dense_zdim")(x)


                z_mu_c = Dense(units=self.z_dim, activation='linear', name="latent_dense_mu_c_{}".format(i))(x)
                #class_cond = Dense(units=len(self.cond_dims), activation='softmax', name="latent_dense_mu_c_class_{}".format(i))(x)

                dim_cond = Dense(units=self.z_dim, activation='hard_sigmoid', name= "latent_dense_mu_c_dim_{}".format(i))(class_cond)

                z_mu_cond = Multiply()([z_mu_c, dim_cond])

                mu_cond.append(z_mu_cond)

        z_mu = Add()([z_mu] + mu_cond)


        return Model(inputs=x_inputs, outputs=[z_mu, z_log_sigma], name='encoder')

    def build_decoder(self):
        """
        Decoder: P(X|z,y)
        :return:
        """

        x_inputs = Input(shape=(self.z_dim,), name='dec_z')
        x = x_inputs
        nLayers=len(self.d_dims)
        for idx, layer_dim in reversed(list(enumerate(self.d_dims))):
            #x = Dense(units=layer_dim, activation='relu', name='dec_dense_{}'.format(idx))(x)
            if(idx==0):
                x = concatenate([Dense(units=layer_dim, activation='relu')(x), x],name="dec_dense_resnet{}".format(idx)) 
                #x = Dense(units=layer_dim, activation='relu',name="dec_dense_resnet{}".format(idx))(x) 
            else:
                if (idx==0 and self.embeddingBeforeLatent):#we make the embedding more influential
                    #x = concatenate([Dense(units=layer_dim, activation='relu')(x), cond_inputs], name="enc_dense_resnet{}".format(idx)) #plus rapide dans l'apprentissage mais sans doute moins scalable..!
                    print('cool')
                    x = concatenate([Dense(units=layer_dim, activation='relu')(x), x],name="dec_dense_resnet{}".format(idx)) 

                else:
                    #x = Dense(units=layer_dim, activation='relu',name="dec_dense_resnet{}".format(idx))(x) 
                    if(self.has_skip):
                        x = concatenate([Dense(units=layer_dim, activation='relu')(x), x],name="dec_dense_resnet{}".format(idx)) 
                    else:
                        x = Dense(units=layer_dim, activation='relu',name="dec_dense_resnet{}".format(idx))(x) 
            #x = Dropout(self.dropout)(x)
        #xprevious=x
        output = Dense(units=self.input_dim[0], activation='linear', name='dec_x_hat')(x)
        #outputBis = Lambda(lambda x: x)(x)

        return Model(inputs=x_inputs, outputs=output, name='decoder')

    def build_loss(self, z_mu, z_log_sigma,beta=0):
        """

        :return:
        """

        def kl_loss(y_true, y_pred):
            if self.prior == 'Gaussian':
                return 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mu) - 1. - z_log_sigma, axis=-1)
            elif self.prior == 'Laplace':
                return K.sum(K.abs(z_mu) + K.exp(z_log_sigma)*K.exp(-K.abs(z_mu)/K.exp(z_log_sigma)) - 1. - z_log_sigma, axis=-1)


        def recon_loss(y_true, y_pred):
            if(self.is_L2_Loss):
                print("L2 loss")
                print(self.is_L2_Loss)
                return K.sum(K.square(y_pred - y_true), axis=-1)
            else:
                print("L1 loss")
                print(self.is_L2_Loss)
                return K.sum(K.abs(y_pred - y_true), axis=-1)

        def vae_loss(y_true, y_pred, beta=0, gamma=0):
            """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """

            # E[log P(X|z,y)]
            recon = recon_loss(y_true=y_true, y_pred=y_pred)

            # D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian
            kl = kl_loss(y_true=y_true, y_pred=y_pred)

            return recon + beta*kl

        return vae_loss, recon_loss, kl_loss
    
    def build_loss_info(self, z_mu, z_log_sigma, z,beta=0.5, gamma=1):
        """

        :return:
        """

        def kl_loss(y_true, y_pred):
            if self.prior == 'Gaussian':
                return 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mu) - 1. - z_log_sigma, axis=-1)
            elif self.prior == 'Laplace':
                return K.sum(K.abs(z_mu) + K.exp(z_log_sigma)*K.exp(-K.abs(z_mu)/K.exp(z_log_sigma)) - 1. - z_log_sigma, axis=-1)


        def recon_loss(y_true, y_pred):
            if(self.is_L2_Loss):
                print("L2 loss")
                print(self.is_L2_Loss)
                return K.sum(K.square(y_pred - y_true), axis=-1)
            else:
                print("L1 loss")
                print(self.is_L2_Loss)
                return K.sum(K.abs(y_pred - y_true), axis=-1)

        def kde(s1,s2,h=None):
            dim = K.shape(s1)[1]
            s1_size = K.shape(s1)[0]
            s2_size = K.shape(s2)[0]
            if h is None:
                h = K.cast(dim, dtype='float32') / 2
            tiled_s1 = K.tile(K.reshape(s1, K.stack([s1_size, 1, dim])), K.stack([1, s2_size, 1]))
            tiled_s2 = K.tile(K.reshape(s2, K.stack([1, s2_size, dim])), K.stack([s1_size, 1, 1]))
            return K.exp(-0.5 * K.sum(K.square(tiled_s1 - tiled_s2), axis=-1)  / h)

        def info_loss(y_true, y_pred):
            q_kernel = kde(z_mu, z_mu)
            p_kernel = kde(z, z)
            pq_kernel = kde(z_mu, z)
            return K.mean(q_kernel) + K.mean(p_kernel) - 2 * K.mean(pq_kernel)

        def vae_loss(y_true, y_pred, beta=0, gamma=0):
            """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """

            # E[log P(X|z,y)]
            recon = recon_loss(y_true=y_true, y_pred=y_pred)

            # D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian
            kl = kl_loss(y_true=y_true, y_pred=y_pred)

            #D(q(z)|| p(z)); calculated with the MMD estimator using a Gaussian kernel
            info = info_loss(y_true=y_true, y_pred=y_pred)

            return recon + beta*kl + gamma*info

        return vae_loss, recon_loss, kl_loss, info_loss


    def train(self, dataset_train, training_epochs=10, batch_size=20, callbacks = [], validation_data = None, verbose = True,validation_split=None):
        """

        :param dataset_train:
        :param training_epochs:
        :param batch_size:
        :param callbacks:
        :param validation_data:
        :param verbose:
        :return:
        """

        assert len(dataset_train) >= 2  # Check that both x and cond are present
        #outputs=np.array([dataset_train['y'],dataset_train['y1']])
        output1=dataset_train['y']
        output2=dataset_train['y']

        if self.InfoVAE:
            output3=dataset_train['y']

            cvae_hist = self.cvae.fit(dataset_train['x'], [output1,output2,output3], batch_size=batch_size, epochs=training_epochs,
                                 validation_data=validation_data,validation_split=validation_split,
                                 callbacks=callbacks, verbose=verbose)
        else:
            cvae_hist = self.cvae.fit(dataset_train['x'], [output1,output2], batch_size=batch_size, epochs=training_epochs,
                                 validation_data=validation_data,validation_split=validation_split,
                                 callbacks=callbacks, verbose=verbose)

        return cvae_hist

