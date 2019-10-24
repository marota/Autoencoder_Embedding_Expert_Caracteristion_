import os
import json
from matplotlib import pyplot as plt
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Dropout, BatchNormalization, Activation
from keras.layers.merge import concatenate, Add, Subtract, Multiply, Concatenate
from keras import backend as K
from keras.callbacks import Callback
from keras import losses
from keras import optimizers
from functools import partial, update_wrapper


class Class_Statistics(Callback):
    def __init__(self, dataset_train, n_class, emb=False):
        self.dataset_train = dataset_train
        self.emb=emb

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 100 == 0:
            lays_enc = self.model.get_layer('encoder')
            
            if self.emb:
                lays_emb = self.model.get_layer('embedding_enc')
                emb_inputs = lays_emb.predict([self.dataset_train[i] for i in range(1,len(self.dataset_train))])
                input_encoder = [self.dataset_train[0],emb_inputs]
                
            else:
                input_encoder = self.dataset_train

            labels = lays_enc.predict(input_encoder)[-1]
            label_w = np.sum(labels, axis=0)/labels.shape[0]
            print(label_w)

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


#un modèle CVAE ou l'on passe un prior sous forme d'un mélange gaussien
class CVAE_mixture(BaseModel):
    def __init__(self, input_dim=96, cond_dim=12, z_dim=2, e_dims=[24], d_dims=[24], beta=1, embeddingBeforeLatent=False,pDropout=0.0, verbose=True,is_L2_Loss=True, InfoVAE = False, gamma=0, pdf_model='Gaussian',prior_mixture=K.variable(np.ones(2)/2,dtype='float32'),prior_mu=K.variable(np.zeros((2,2)),dtype='float32'),sigma_prior=None,has_skip=True,has_BN=1, lr = 0.001,**kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.cond_dim = cond_dim
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
        self.pdf_model = pdf_model #distribution model for prior and posterior
        if sigma_prior is not None:
            self.log_sigma_prior = K.log(K.variable(sigma_prior, dtype='float32')) #form of the diagonal matrix of covariance for all components of the mixture model
        else:
            self.log_sigma_prior = K.zeros(self.z_dim, dtype='float32')
        assert K.eval(K.sum(prior_mixture)) == 1.
        assert K.int_shape(prior_mu)[0] == z_dim
        self.prior_mixture = prior_mixture #distribution of weights for the mixture model
        self.prior_mu = prior_mu #array of mean values for the each component of the mixture model
        """
        self.entropy_loss = entropy_loss
        if entropy_loss:
            self.h = h
            self.alpha = alpha
        """

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

        x_true = Input(shape=(self.input_dim,), name='x_true')
        cond_true = Input(shape=(self.cond_dim,), name='cond_pre')

        # Encoding
        z_mu, z_log_sigma = self.encoder([x_true, cond_true])
        #self.latent=Lambda(lambda x:x,'latent')(z_mu)
        
        x_inputs = Input(shape=(self.input_dim,), name='x_true_zmu_Layer') 
        x = Lambda(lambda x: x,name='z_mu')(x_inputs)
        self.latent=Model(inputs=[x_inputs], outputs=[x], name='z_mu_output')
        ZMU=self.latent(z_mu)

        # Sampling
        # Here if the prior is Gaussian, z_log_sigma is the log of sigma², whereas it refers to the log of sigma if it's Laplacian. 
        def sample_z(args):
            mu, log_sigma = args
            if self.pdf_model=='Gaussian':
                eps = K.random_normal(shape=(K.shape(mu)[0], self.z_dim), mean=0., stddev=1.)
                return mu + K.exp(log_sigma / 2) * eps
            elif self.pdf_model=='Laplace':
                U = K.random_uniform(shape=(K.shape(mu)[0], self.z_dim), minval =0.0, maxval=1.)
                V = K.random_uniform(shape=(K.shape(mu)[0], self.z_dim), minval =0.0, maxval=1.)
                Rad_sample = 2.*K.cast(K.greater_equal(V,0.5), dtype='float32') - 1. 
                Expon_sample = -K.exp(log_sigma)*K.log(1-U)
                return mu + Rad_sample*Expon_sample

        z = Lambda(sample_z, name='sample_z')([z_mu, z_log_sigma])

        # Decoding
        x_hat= self.decoder([z, cond_true])
        
        #identity layer to have two output layers and compute separately 2 losses (the kl and the reconstruction)
             
        x = Lambda(lambda x: x)(x_inputs)
        identitModel=Model(inputs=[x_inputs], outputs=[x], name='decoder_for_kl')
        if self.InfoVAE :
            identitModel2=Model(inputs=[x_inputs], outputs=[x], name='decoder_info')
            xhatTer = identitModel2(x_hat)
        """
        if self.entropy_loss:
            identitModel3=Model(inputs=[x_inputs], outputs=[x], name='cond_mi')
            xhatQua = identitModel3(x_hat)
        """
       
        xhatBis=identitModel(x_hat)

        # Defining loss
        if self.InfoVAE:
            # Defining loss
            vae_loss, recon_loss, kl_loss, info_loss= self.build_loss_info(z_mu, z_log_sigma, z, beta=self.beta, gamma=self.gamma)

            # Defining and compiling cvae model
            self.losses = {"decoder": recon_loss,"decoder_for_kl": kl_loss, "decoder_info": info_loss}
            #lossWeights = {"decoder": 1.0, "decoder_for_kl": 0.01}
            self.weight_losses = {"decoder": 1.0, "decoder_for_kl": self.beta, "decoder_info":self.gamma}
            model_outputs = [x_hat,xhatBis, xhatTer]

        else:
            vae_loss, recon_loss, kl_loss = self.build_loss(z_mu, z_log_sigma, beta=self.beta)

            # Defining and compiling cvae model
            self.losses = {"decoder": recon_loss,"decoder_for_kl": kl_loss}
            #lossWeights = {"decoder": 1.0, "decoder_for_kl": 0.01}
            self.weight_losses = {"decoder": 1.0, "decoder_for_kl": self.beta}
            model_outputs = [x_hat,xhatBis]
        
        """
        if self.entropy_loss:
            
            model_outputs = model_outputs + [xhatQua]
            entropy_losses = self.build_entropy_loss(z_mu, cond_true, x_true, recon_loss)
            self.losses.update({"cond_mi" : entropy_losses})
            self.weight_losses.update({"cond_mi" :1.0})
        """

        Opt_Adam = optimizers.Adam(lr=self.lr)
        
        self.cvae = Model(inputs=[x_true, cond_true], outputs=model_outputs)#self.encoder.outputs])
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
        x_inputs = Input(shape=(self.input_dim,), name='enc_x_true')
        
        if(self.cond_dim>=1):

            cond_inputs = Input(shape=(self.cond_dim,), name='enc_cond')
            x = concatenate([x_inputs, cond_inputs], name='enc_input')
        else:
            cond_inputs = Input(shape=(0,), name='enc_cond')
            x = concatenate([x_inputs, cond_inputs], name='enc_input')

        nLayers = len(self.e_dims)
        for idx, layer_dim in enumerate(self.e_dims):
            #x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
            if (idx<(nLayers-1)):
                x = concatenate([Dense(units=layer_dim, activation='relu')(x),cond_inputs], name="enc_dense_{}".format(idx))
                #x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
            else:
                #x = Dense(units=layer_dim, activation='sigmoid', name="enc_dense_{}".format(idx))(x)
                x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
            #x = Dropout(self.dropout)(x)

        #z_mu = Dense(units=self.z_dim, activation='linear', name="latent_dense_mu")(x)
        #z_log_sigma = Dense(units=self.z_dim, activation='linear', name='latent_dense_log_sigma')(x)
        #x = Dense(units=self.z_dim, activation='relu', name="enc_dense_zdim")(x)
        z_mu = Dense(units=self.z_dim, activation='linear', name="latent_dense_mu")(x)
        z_log_sigma = Dense(units=self.z_dim, activation='linear', name='latent_dense_log_sigma')(x)

        return Model(inputs=[x_inputs, cond_inputs], outputs=[z_mu, z_log_sigma], name='encoder')

    def build_decoder(self):
        """
        Decoder: P(X|z,y)
        :return:
        """

        x_inputs = Input(shape=(self.z_dim,), name='dec_z')
        
        if(self.cond_dim>=1):
            cond_inputs = Input(shape=(self.cond_dim,), name='dec_cond')
            x = concatenate([x_inputs, cond_inputs], name='dec_input')#BatchNormalization()(cond_inputs) or not?
        else:
            cond_inputs = Input(shape=(0,), name='dec_cond')
            x = concatenate([x_inputs, cond_inputs], name='dec_input')

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
        output = Dense(units=self.input_dim, activation='linear', name='dec_x_hat')(x)
        #outputBis = Lambda(lambda x: x)(x)

        return Model(inputs=[x_inputs, cond_inputs], outputs=output, name='decoder')


    def build_loss(self, z_mu, z_log_sigma,beta=0):
        """

        :return:
        """

        def kl_loss(y_true, y_pred):

            n_mixture = K.int_shape(self.prior_mixture)[0]
            prior_mixture = K.tile(K.reshape(self.prior_mixture, K.stack([1,n_mixture])), K.stack([K.shape(z_mu)[0],1]))        
            log_prior_sigma = K.tile(K.reshape(self.log_sigma_prior, K.stack([1,self.z_dim,1])), K.stack([K.shape(z_mu)[0],1,n_mixture]))
            prior_mu = K.cumsum(K.ones(n_mixture, dtype='float32')) *2. - K.cast(n_mixture, dtype='float32')
            prior_mu = K.tile(K.reshape(prior_mu, K.stack([1,1,n_mixture])), K.stack([K.shape(z_mu)[0],self.z_dim,1]))

            z_extend = K.tile(K.reshape(z_mu, K.stack([K.shape(z_mu)[0],self.z_dim,1])), K.stack([1,1,n_mixture]))
            log_sigma_extend = K.tile(K.reshape(z_log_sigma, K.stack([K.shape(z_mu)[0],self.z_dim,1])), K.stack([1,1,n_mixture]))

            if self.pdf_model == 'Gaussian':
                return K.sum(0.5 * K.sum((K.exp(log_sigma_extend) + K.square(z_extend - prior_mu))/K.exp(log_prior_sigma) - 1. - log_sigma_extend + log_prior_sigma, axis=1) * prior_mixture, axis=-1)
            elif self.pdf_model == 'Laplace':
                return K.sum(K.sum((K.abs(z_extend - prior_mu) + K.exp(log_sigma_extend)*K.exp(-K.abs(z_extend - prior_mu)/K.exp(log_sigma_extend)))/K.exp(log_prior_sigma) - 1. - log_sigma_extend + log_prior_sigma, axis=1) * prior_mixture, axis=-1)


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
            n_mixture = K.int_shape(self.prior_mixture)[0]
            prior_mixture = K.tile(K.reshape(self.prior_mixture, K.stack([1,n_mixture])), K.stack([K.shape(z_mu)[0],1]))        
            log_prior_sigma = K.tile(K.reshape(self.log_sigma_prior, K.stack([1,self.z_dim,1])), K.stack([K.shape(z_mu)[0],1,n_mixture]))
            prior_mu = K.tile(K.reshape(self.prior_mu, K.stack([1,self.z_dim,n_mixture])), K.stack([K.shape(z_mu)[0],1,1]))

            z_extend = K.tile(K.reshape(z_mu, K.stack([K.shape(z_mu)[0],self.z_dim,1])), K.stack([1,1,n_mixture]))
            log_sigma_extend = K.tile(K.reshape(z_log_sigma, K.stack([K.shape(z_mu)[0],self.z_dim,1])), K.stack([1,1,n_mixture]))

            if self.pdf_model == 'Gaussian':
                return K.sum(0.5 * K.sum((K.exp(log_sigma_extend) + K.square(z_extend - prior_mu))/K.exp(log_prior_sigma) - 1. - log_sigma_extend + log_prior_sigma, axis=1) * prior_mixture, axis=-1)
            elif self.pdf_model == 'Laplace':
                return K.sum(K.sum((K.abs(z_extend - prior_mu) + K.exp(log_sigma_extend)*K.exp(-K.abs(z_extend - prior_mu)/K.exp(log_sigma_extend)))/K.exp(log_prior_sigma) - 1. - log_sigma_extend + log_prior_sigma, axis=1) * prior_mixture, axis=-1)


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

    """
    def build_entropy_loss(self, z_mu, cond_true, x_true, recon_loss,h=4.,alpha=1.01):

        def build_gram_matrix(s1,h):
            K.cast(s1, dtype='float64')
            K.cast(h, dtype='float64')

            dim = K.shape(s1)[1]
            s1_size = K.shape(s1)[0]

            tiled_s1 = K.tile(K.reshape(s1, K.stack([s1_size, 1, dim])), K.stack([1, s1_size, 1]))
            tiled_s2 = K.tile(K.reshape(s1, K.stack([1, s1_size, dim])), K.stack([s1_size, 1, 1]))
            Dist_M = K.exp(-0.5 * K.square(K.cast((tiled_s1 - tiled_s2),dtype='float64')  / h))

            list_dist_M = [K.squeeze(K.slice(Dist_M, [0,0,i],K.stack([s1_size,s1_size,1])), axis=-1) for i in range(K.int_shape(s1)[1])]

            gram_list = []
            for dist_M in list_dist_M:
                diag = tf.diag_part(dist_M)
                diag_1 = K.tile(K.reshape(diag, K.stack([s1_size, 1])), K.stack([1, s1_size]))
                diag_2 = K.tile(K.reshape(diag, K.stack([1, s1_size])), K.stack([s1_size, 1]))
                gram_list.append(dist_M / K.sqrt(diag_1*diag_2) / K.cast(s1_size, dtype='float64'))

            return gram_list

        def build_joint_entropy(list_M):

            AB = K.cast(K.ones_like(list_M[0]), dtype='float64')
            for M in list_M:
                AB = AB * M

            res = AB / K.cast(K.reshape(tf.trace(AB),K.stack([1])), dtype='float64')
            return K.reshape(res,K.shape(list_M[0]))

        def build_Renyi_entropy(A,alpha):
            A = K.cast(A, dtype='float64')
            #A = K.switch(K.sum(K.cast(tf.is_nan(A),dtype='float64')) >0, K.ones(1, dtype='float64'), A)
            input_A = K.cast(0.5*(A+K.transpose(A)), dtype='float64')
            eig_values,_ = tf.linalg.eigh(input_A)
            eig_values = eig_values + 1.e-6
            return K.cast(K.log(K.sum(K.pow(eig_values,alpha))), dtype='float64') / K.cast((1. - alpha) * K.log(2.), dtype='float64')

        def entropy_regularizer(y_true, y_pred):
            z = K.switch(K.greater(recon_loss(y_true,y_pred),100.),K.random_normal(K.shape(z_mu), seed=42), z_mu)
            sigma = h * K.pow(K.cast(K.shape(y_true)[0], dtype='float64'), -1./(4.+1.))
            gram_x = build_gram_matrix(x_true,sigma)
            gram_c = build_gram_matrix(cond_true,sigma)
            gram_z = build_gram_matrix(z,sigma) #K.ones_like(z_mu, dtype='float32')
            #gram_z = [K.eye(self.batch_size, dtype='float64')*K.cast(self.buffer < 10000, dtype='float64')]*4 #+ K.cast(self.buffer > 10000, dtype='float64')*g_z for g_z in gram_z] #]

            joint_entropy_x = build_joint_entropy(gram_x)
            joint_entropy_z = build_joint_entropy(gram_z)
            joint_entropy_cond = build_joint_entropy(gram_c)
            joint_entropy_xz = build_joint_entropy(gram_z+gram_x)
            joint_entropy_zcond = build_joint_entropy(gram_z+gram_c)

            cond_MI = build_Renyi_entropy(joint_entropy_z,alpha) + build_Renyi_entropy(joint_entropy_cond,alpha) - build_Renyi_entropy(joint_entropy_zcond, alpha)
            data_MI = build_Renyi_entropy(joint_entropy_z,alpha) + build_Renyi_entropy(joint_entropy_x,alpha) - build_Renyi_entropy(joint_entropy_xz, alpha)

            return K.cast(cond_MI - data_MI, dtype='float32')

        return entropy_regularizer
    """

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
        output_total =[output1] * len(self.losses.keys())

        cvae_hist = self.cvae.fit(dataset_train['x'], output_total, batch_size=batch_size, epochs=training_epochs,
                             validation_data=validation_data,validation_split=validation_split,
                             callbacks=callbacks, verbose=verbose)

        return cvae_hist

#un modèle CVAE où l'on encode la classe plutôt que la projection. On déduit la projection de la classe. 
class CVAE_cluster(BaseModel):
    def __init__(self, dataset_train, input_dim=96, cond_dim=12, z_dim=2, e_dims=[24], d_dims=[24], beta=1,embeddingBeforeLatent=False,pDropout=0.0, verbose=True,is_L2_Loss=True, proba_mixture=np.ones(2)/2,has_skip=True,has_BN=1, lr = 0.001,**kwargs):
        super().__init__(**kwargs)

        self.dataset_train = dataset_train
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.z_dim = z_dim
        self.e_dims = e_dims
        self.d_dims = d_dims
        self.beta = beta
        self.dropout = pDropout#la couche de dropout est pour l'instant commentée car pas d'utilité dans les experiences
        self.encoder = None
        self.decoder = None
        self.latent=None
        self.cvae = None
        self.emb_dims = None
        self.embeddingBeforeLatent=embeddingBeforeLatent#in the decoder, do a skip only with the embedding and not the latent space to make it more influencial
        self.verbose = verbose
        self.losses={}
        self.weight_losses={}
        self.is_L2_Loss=is_L2_Loss
        self.has_skip=has_skip
        self.lr=lr
        self.has_BN=has_BN
        self.proba_mixture = proba_mixture #array of prior distribution of each cluster weight in the dataset
        self.n_clusters=len(proba_mixture)
        

        self.build_model()

    def build_model(self):
        """

        :param verbose:
        :return:
        """

        self.encoder= self.build_encoder()
        self.decoder = self.build_decoder()
        
        x_true = Input(shape=(self.input_dim,), name='x_true')
        cond_true = Input(shape=(self.cond_dim,), name='cond_pre')

        # Encoding

        z_mu, z_log_sigma, z_y = self.encoder([x_true, cond_true])

        x_inputs = Input(shape=(self.input_dim,), name='x_true_zmu_Layer') 
        x = Lambda(lambda x: x,name='z_mu')(x_inputs)
        self.latent=Model(inputs=[x_inputs], outputs=[x], name='z_mu_output')
        ZMU=self.latent(z_mu)

        # Decoding
        x_hat, prior_mu, prior_log_sigma = self.decoder([z_y, cond_true])

        list_labels_mu=[]
        list_labels_log_sigma=[]
        for i in range(self.n_clusters):
            label_tensor = K.one_hot(K.cast(i * K.ones(1), dtype='int32'), self.n_clusters)
            _,c_prior_mu,c_prior_log_sigma = self.decoder([label_tensor,cond_true])
            list_labels_mu.append(c_prior_mu)
            list_labels_log_sigma.append(c_prior_log_sigma)

        #identity layer to have two output layers and compute separately 2 losses (the kl and the reconstruction)
             
        x = Lambda(lambda x: x)(x_inputs)
        identitModel=Model(inputs=[x_inputs], outputs=[x], name='decoder_for_kl')
        
        xhatBis=identitModel(x_hat)

        # Defining loss
        vae_loss, recon_loss, kl_loss = self.build_loss_mixture(z_mu, z_log_sigma, list_labels_mu, list_labels_log_sigma,weight=self.beta)

        # Defining and compiling cvae model
        self.losses = {"decoder": recon_loss,"decoder_for_kl": kl_loss}
        #lossWeights = {"decoder": 1.0, "decoder_for_kl": 0.01}
        self.weight_losses = {"decoder": 1.0, "decoder_for_kl": self.beta}

        Opt_Adam = optimizers.Adam(lr=self.lr)
        
        if(self.cond_dim==0):
            self.cvae = Model(inputs=[x_true, cond_true], outputs=[x_hat,xhatBis])#self.encoder.outputs])
            #self.cvae.compile(optimizer='rmsprop', loss=vae_loss, metrics=[kl_loss, recon_loss])
            self.cvae.compile(optimizer=Opt_Adam,loss=self.losses,loss_weights=self.weight_losses)
        else:
            self.cvae = Model(inputs=[x_true, cond_true], outputs=[x_hat,xhatBis])#self.encoder.outputs])
            #self.cvae.compile(optimizer='Adam', loss=vae_loss, metrics=[kl_loss, recon_loss])
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
        x_inputs = Input(shape=(self.input_dim,), name='enc_x_true')
        if(self.cond_dim>=1):
            cond_inputs = Input(shape=(self.cond_dim,), name='enc_cond_y')
        else:
            cond_inputs = Input(shape=(0,), name='enc_cond_y')

        nLayers = len(self.e_dims)

        #création de la variable catégorielle
        x1 = concatenate([x_inputs, cond_inputs], name='enc_input_y')

        for idx, layer_dim in enumerate(self.e_dims):
            #x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
            if (idx<(nLayers-1)):
                x1 = concatenate([Dense(units=layer_dim, activation='relu')(x1),cond_inputs], name="enc_cat_dense_{}".format(idx))
                #x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
            else:
                #x = Dense(units=layer_dim, activation='sigmoid', name="enc_dense_{}".format(idx))(x)
                x1 = Dense(units=layer_dim, activation='relu', name="enc_cat_dense_{}".format(idx))(x1)
            #x = Dropout(self.dropout)(x)
        z_y = Dense(units=3*self.n_clusters, activation='relu', name="latent_dense_pre_y")(x1)

        def class_reg(y_outputs):
            return 10 * (1 - K.sqrt(K.sum(K.square(y_outputs)))) 

        z_y = Dense(units=self.n_clusters, activation='linear', name="latent_dense_y")(x1)
        z_y = Lambda(lambda y: K.softmax(10000*(y+1e-8)), name = 'latent_y')(z_y)
        #z_y = Dense(units=self.n_clusters, activation='softmax', name="latent_y")(z_y) #, activity_regularizer = class_reg
        #z_label = Lambda(lambda y: K.ones(1) * K.cast(K.argmax(y, axis=-1), dtype='float32'), name='latent_labels')(z_y)
        def one_hot_cat(z_y):
            one_hot_labels = K.one_hot(K.cast(z_label, dtype='int32'), self.n_clusters)
            return one_hot_labels

        #z_cat = Lambda(one_hot_cat, name='latent_cat')(z_label)

        #création des représentations
        
        x = concatenate([x_inputs, cond_inputs, z_y], name='enc_input')
        for idx, layer_dim in enumerate(self.e_dims):
            #x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
            if (idx<(nLayers-1)):
                x = concatenate([Dense(units=layer_dim, activation='relu')(x),cond_inputs, z_y], name="enc_dense_{}".format(idx))
                #x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
            else:
                #x = Dense(units=layer_dim, activation='sigmoid', name="enc_dense_{}".format(idx))(x)
                x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
            #x = Dropout(self.dropout)(x)

            #z_mu = Dense(units=self.z_dim, activation='linear', name="latent_dense_mu")(x)
            #z_log_sigma = Dense(units=self.z_dim, activation='linear', name='latent_dense_log_sigma')(x)
            #x = Dense(units=self.z_dim, activation='relu', name="enc_dense_zdim")(x)
        z_mu = Dense(units=self.z_dim, activation='linear', name="latent_dense_mu")(x)
        z_log_sigma = Dense(units=self.z_dim, activation='linear', name='latent_dense_log_sigma')(x)


        if(self.cond_dim>=1):
            return Model(inputs=[x_inputs, cond_inputs], outputs=[z_mu, z_log_sigma, z_y], name='encoder')
        else:
            return Model(inputs=[x_inputs, cond_inputs], outputs=[z_mu, z_log_sigma, z_y], name='encoder')

    def build_decoder_cat(self):
        """
        Encoder: p(z|y)
        :return:
        """
        #x_inputs = Input(shape=(self.n_clusters,), name='prior_x_true')
        x_inputs = Input(shape=(self.n_clusters,), name='prior_x_true')

        nLayers = len(self.d_dims)
        x = Dense(units=self.e_dims[-1], activation='relu', name="dec_lat_dense_1")(x_inputs)
        x = concatenate([Dense(units=self.e_dims[-1], activation='relu')(x),x], name="dec_lat_dense_2")

        #x = x_inputs
        #for idx, layer_dim in enumerate([self.n_clusters, self.n_clusters, self.e_dims[-1]]):
            #x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
            #x = Dense(units=layer_dim, activation='relu', name="prior_dense_{}".format(idx))(x)
            #x = Dropout(self.dropout)(x)

        z_mu = Dense(units=self.z_dim, activation='linear', name="prior_dense_mu")(x)
        z_log_sigma = Dense(units=self.z_dim, activation='linear', name='prior_dense_log_sigma')(x)
        #z_mu = Lambda(lambda x: K.ones_like(x[0])*K.tile(K.reshape(K.cast(K.argmax(x[1], axis=-1), dtype='float32'),shape=(-1,1)), K.stack([1,K.shape(x[0])[1]])), name="prior_dense_mu")([z_log_sigma, x_inputs])


        if(self.cond_dim>=1):
            return Model(inputs=x_inputs, outputs=[z_mu, z_log_sigma], name='decoder_cat')
        else:
            return Model(inputs=x_inputs, outputs=[z_mu, z_log_sigma], name='decoder_cat')

    def build_decoder_latent(self):
        """
        Decoder_latent: P(X|z)
        :return:
        """

        x_inputs = Input(shape=(self.z_dim,), name='dec_z')
        
        if(self.cond_dim>=1):
            cond_inputs = Input(shape=(self.cond_dim,), name='dec_cond')
        else:
            cond_inputs = Input(shape=(0,), name='dec_cond')

        x = concatenate([x_inputs, cond_inputs], name='dec_input')  

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
        output = Dense(units=self.input_dim, activation='linear', name='dec_x_hat')(x)
        #outputBis = Lambda(lambda x: x)(x)

        if(self.cond_dim>=1):
            return Model(inputs=[x_inputs, cond_inputs], outputs=output, name='decoder_latent')
        else:
            return Model(inputs=[x_inputs, cond_inputs], outputs=output, name='decoder_latent')


    def build_decoder(self):

        self.decoder_cat = self.build_decoder_cat()
        self.decoder_latent = self.build_decoder_latent()

        x_inputs = Input(shape=(self.n_clusters,), name='dec_x_true')
        #x_inputs = Input(shape=(self.n_clusters,), name='dec_x_true')
        if(self.cond_dim>=1):
            cond_inputs = Input(shape=(self.cond_dim,), name='dec_cond')
        else:
            cond_inputs = Input(shape=(0,), name='dec_cond')

        prior_mu, prior_log_sigma = self.decoder_cat(x_inputs)

        def sample_z(args):
            mu_mix, log_sigma_mix = args
            #sampling per category
            eps = K.random_normal(shape=(K.shape(mu_mix)[0], self.z_dim), mean=0., stddev=1.)
            z_samples=mu_mix + K.exp(log_sigma_mix / 2) * eps
            return z_samples   

        z = Lambda(sample_z, name='sample_z')([prior_mu, prior_log_sigma])

        # Decoding
        x_hat= self.decoder_latent([z, cond_inputs])

        if(self.cond_dim>=1):
            return Model(inputs=[x_inputs, cond_inputs], outputs=[x_hat, prior_mu, prior_log_sigma], name='decoder')
        else:
            return Model(inputs=[x_inputs, cond_inputs], outputs=[x_hat, prior_mu, prior_log_sigma], name='decoder')
        return Model(inputs=cat_input, outputs=label_proba, name='label_weight_display')


    def build_loss_mixture(self, z_mu, z_log_sigma,list_labels_mu, list_labels_log_sigma,weight=0):
        
        def kl_loss(y_true, y_pred):
            labels_weights = K.get_value(K.sum(self.encoder.predict(self.dataset_train)[-1], axis=0)/ K.cast(K.shape(self.dataset_train[0])[0], dtype='float32'))
            
            distribution_loss = 0.5* K.sum([(labels_weights[k]+1e-8) * K.sum(list_labels_log_sigma[k] + (K.exp(z_log_sigma) + K.square(z_mu - 1000*list_labels_mu[k])) / K.exp(list_labels_log_sigma[k]) - 1. - z_log_sigma, axis=-1) for k in range(self.n_clusters)])

            category_loss = K.cast(K.sum(-K.log(self.proba_mixture/(labels_weights+1e-8))*labels_weights, axis=-1), dtype='float32')

            return category_loss + distribution_loss


        def recon_loss(y_true, y_pred):
            if(self.is_L2_Loss):
                print("L2 loss")
                print(self.is_L2_Loss)
                return K.sum(K.square(y_pred - y_true), axis=-1)
            else:
                print("L1 loss")
                print(self.is_L2_Loss)
                return K.sum(K.abs(y_pred - y_true), axis=-1)

        def vae_loss(y_true, y_pred,weight=0):
            """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """

            # E[log P(X|z,y)]
            recon = recon_loss(y_true=y_true, y_pred=y_pred)

            # D_KL(Q(z|X,y) || P(z|X)); calculate in closed form as both dist. are Gaussian
            kl = kl_loss(y_true=y_true, y_pred=y_pred)

            return recon + weight*kl 

        return vae_loss, recon_loss, kl_loss

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

        cvae_hist = self.cvae.fit(dataset_train['x'], [output1,output2], batch_size=batch_size, epochs=training_epochs,
                             validation_data=validation_data,validation_split=validation_split,
                             callbacks=callbacks, verbose=verbose)

        return cvae_hist

#un modèle CVAE ou l'on passe les conditions avec un meme embedding avant d etre passe en entrée ou dans l'espace latent
class CVAE_emb(CVAE_cluster):
    """
    Improvement of CVAE that encode the temperature as a condition
    """
    def __init__(self, to_emb_dim=96, cond_pre_dim=12, emb_dims=[2], emb_to_z_dim=[3],is_emb_Enc_equal_emb_Dec=True, **kwargs):

        self.to_emb_dim = to_emb_dim
        self.cond_pre_dim = cond_pre_dim
        self.emb_dims = emb_dims
        self.emb_to_z_dim=emb_to_z_dim
        self.embedding_enc = None
        self.embedding_dec = None
        self.is_emb_Enc_equal_emb_Dec=is_emb_Enc_equal_emb_Dec
        
        cond_dim=self.cond_pre_dim 
        if(len(self.emb_to_z_dim)>=1):
            cond_dim=self.cond_pre_dim + self.emb_to_z_dim[-1]
        print(cond_dim)
        super().__init__(cond_dim=cond_dim,**kwargs)

    def build_model(self):
        """

        :param verbose:
        :return:
        """

        self.encoder = self.build_encoder()
        self.encoder_cat = self.build_encoder_cat()
        self.decoder = self.build_decoder()
        
        if(len(self.emb_to_z_dim)>=1):
            self.embedding_enc = self.build_embedding(name_emb='embedding_enc')
            if(self.is_emb_Enc_equal_emb_Dec):
                self.embedding_dec = self.embedding_enc
            else:
                self.embedding_dec = self.build_embedding(name_emb='embedding_dec')

        x_true = Input(shape=(self.input_dim,), name='x_true')
        
        inputs=[x_true]
        xembs=[]
        cond_pre=[]
        if(self.cond_pre_dim>=1):
            cond_pre = Input(shape=(self.cond_pre_dim,), name='cond_pre')
            inputs.append(cond_pre)
        for j, cond in enumerate(self.to_emb_dim):#on enumere sur les conditions
            to_emb_dim=self.to_emb_dim[j]
            x_input = Input(shape=(to_emb_dim,), name='emb_input_{}'.format(j))
            xembs.append(x_input)
            inputs.append(x_input)
        
        cond_true_enc=[]
        cond_true_dec=[]#meme embedding que cond_true_enc en l etat
        if(len(self.emb_to_z_dim)>=1):
            cond_emb = self.embedding_enc(xembs)
            cond_true_enc =cond_emb 
            cond_emb2 = self.embedding_dec(xembs)
            if(self.is_emb_Enc_equal_emb_Dec):
                cond_true_dec =cond_emb 
            else:
                print('enc different de dec')
                cond_true_dec=cond_emb2
        if((self.cond_pre_dim>=1) and (len(self.emb_to_z_dim)>=1)):
            cond_true_enc = concatenate([cond_pre, cond_emb], name='conc_cond_enc')
            if(self.is_emb_Enc_equal_emb_Dec):
                cond_true_dec = concatenate([cond_pre, cond_emb], name='conc_cond_dec')
            else:
                print('enc different de dec 2')
                cond_true_dec = concatenate([cond_pre, cond_emb2], name='conc_cond_dec')
        elif(self.cond_pre_dim>=1):
            cond_true=cond_pre

        # Encoding
        z_mu, z_mix_mu, z_mix_log_sigma, z_y = self.encoder([x_true, cond_true_enc])
        
        x_inputs = Input(shape=(self.input_dim,), name='x_true_zmu_Layer') 
        x = Lambda(lambda x: x,name='z_mu')(x_inputs)
        self.latent=Model(inputs=[x_inputs], outputs=[x], name='z_mu_output')
        ZMU=self.latent(z_mu)

        if self.prior != 'Mixture':
            Model(inputs=[x_inputs], outputs=[z_mu], name='encoder')
        else:
            Model(inputs=[x_inputs], outputs=[z_mu, z_y], name='encoder')

        # Sampling
        def mixture_z(z_y, mix_mu):
            y_expand = K.expand_dims(z_y, axis=-1)
            one_M = K.ones(shape=(1,self.z_dim))
            mult_coef = K.permute_dimensions(K.dot(y_expand,one_M), pattern=(0, 2, 1))
            z_mu = K.sum(z_mix_mu * mult_coef, axis=-1)
            return z_mu

        def sample_z(args):
            y, mu_mix, log_sigma_mix = args
            #sampling per category
            eps = K.random_normal(shape=(K.shape(mu_mix)[0], self.z_dim, self.n_clusters), mean=0., stddev=1.)
            z_samplings=mu_mix + K.exp(log_sigma_mix / 2) * eps
            z_samples = self.mixture_z(y, mu_mix)
            return z_samples   


        z = Lambda(sample_z, name='sample_z')([z_y, z_mix_mu, z_mix_log_sigma])

        # Decoding
        x_hat = self.decoder([z, cond_true_dec])
        
        #identity layer to have two output layers and compute separately 2 losses (the kl and the reconstruction)
        x_inputs = Input(shape=(self.input_dim,), name='x_true_identity_Layer')         
        x = Lambda(lambda x: x)(x_inputs)
        identitModel=Model(inputs=[x_inputs], outputs=[x], name='decoder_for_kl')
        
        xhatBis=identitModel(x_hat)

        # Defining loss
        vae_loss, recon_loss, kl_loss = self.build_loss_mixture(z_mix_mu, z_mix_log_sigma, z_y,weight=self.beta)

        # Defining and compiling cvae model
        self.losses = {"decoder": recon_loss,"decoder_for_kl": kl_loss}
        #lossWeights = {"decoder": 1.0, "decoder_for_kl": 0.01}
        self.weight_losses = {"decoder": 1.0, "decoder_for_kl": self.beta}
        
        self.cvae = Model(inputs=inputs, outputs=[x_hat,xhatBis])

        Opt_Adam = optimizers.Adam(lr=self.lr)
        self.cvae.compile(optimizer=Opt_Adam,loss=self.losses,loss_weights=self.weight_losses, metrics=[kl_loss, recon_loss])#, metrics=[kl_loss, recon_loss])

        # Store trainers
        self.store_to_save('cvae')

        self.store_to_save('encoder')
        self.store_to_save('decoder')

        if self.verbose:
            print("complete model: ")
            self.cvae.summary()
            print("embedding_enc: ")
            if(len(self.emb_to_z_dim)>=1):
                self.embedding_enc.summary()
            print("encoder: ")
            self.encoder.summary()
            print("decoder: ")
            self.decoder.summary()
    

    def build_embedding(self,name_emb='embedding'):
        """
        Embedding of the temperature
        :return:
        """

        #verifier que les dimensions des inputs sont cohérentes
        if(len(self.emb_dims)!=len(self.to_emb_dim) ):
            print("dimensions du nombre de conditions dans les embeddings incoherent")
        xinputs=[]
        embeddings=[]
        x_input=[]
        
        for j, cond in enumerate(self.emb_dims):#on enumere sur les conditions
            to_emb_dim=self.to_emb_dim[j]
            x_input = Input(shape=(to_emb_dim,), name="emb_input_{}".format(j))
            xinputs.append(x_input)
            x = x_input
            
            nLayersCond=len(cond)
            for idx, layer_dim in enumerate(cond):
                if(idx==nLayersCond-1):
                    x = Dense(units=layer_dim, activation=None, name="emb_dense_{}_{}".format(j,idx))(x)
                    
                    ###############
                    if(self.has_BN==2):
                        x = BatchNormalization()(x)
                    ##############
                    x = Activation('relu')(x)
                else:
                    x = Dense(units=layer_dim, activation='relu', name="emb_dense_{}_{}".format(j,idx))(x)  
            if not cond:
                x = Dense(units=to_emb_dim, activation='linear', name="emb_linear_{}".format(j))(x)
            embeddings.append(x)
        
        if(len(xinputs)>=2):
            embedding_last= concatenate(embeddings, name='emb_concat')
        else:
            embedding_last=embeddings[0]  
        #embedding = Dense(units=self.emb_dims[-1], activation='linear', name="emb_dense_last")(x)

        emb_size=embedding_last.get_shape()[1]
        
        firstEmbDim=self.emb_to_z_dim[0]
        #if(emb_size>firstEmbDim):
        for j, layer_dim in enumerate(self.emb_to_z_dim):#on enumere sur les conditions
            embedding_last = Dense(units=layer_dim, activation=None,name="emb_dense_last_reduction_{}".format(j))(embedding_last)
               
            ######################
            if(self.has_BN>=1):
                embedding_last = BatchNormalization()(embedding_last)
            ############
               
            #embedding_last = Activation('relu')(embedding_last)
            embedding_last = Activation('relu')(embedding_last)
        
        model=Model(inputs=xinputs, outputs=embedding_last, name=name_emb)
        
        return model
    
    def freezeLayers(self,mondule_names=['encoder']):
        
        if('encoder' in mondule_names):
            for layer in self.encoder.layers:
                layer.trainable = False
        if('decoder' in mondule_names):
            for layer in self.decoder.layers:
                layer.trainable = False
        if('embedding_enc' in mondule_names):
            print('embedding_enc')
            for layer in self.embedding_enc.layers:
                layer.trainable = False
                print(layer.name)
        if('embedding_dec' in mondule_names):
            print('embedding_dec')
            for layer in self.embedding_dec.layers:
                layer.trainable = False
                print(layer.name)
        self.cvae.compile(optimizer='Adam',loss=self.losses,loss_weights=self.weight_losses)
                
    def unfreezeLayers(self,mondule_names=['encoder']):
        
        if('encoder' in mondule_names):
            input_names=self.encoder.input_names
            for layer in self.encoder.layers:
                if(layer.name not in input_names):
                    layer.trainable = True
                    
        if('decoder' in mondule_names):
            input_names=self.decoder.input_names
            for layer in self.decoder.layers:
                if(layer.name not in input_names):
                    layer.trainable = True
                
        if('embedding_enc' in mondule_names):
            input_names=self.embedding_enc.input_names
            for layer in self.embedding_enc.layers:
                if(layer.name not in input_names):
                    layer.trainable = True
        
        if('embedding_dec' in mondule_names):
            input_names=self.embedding_dec.input_names
            for layer in self.embedding_dec.layers:
                if(layer.name not in input_names):
                    layer.trainable = True
        
        self.cvae.compile(optimizer='Adam',loss=self.losses,loss_weights=self.weight_losses)
        
    def updateLossWeight(self,newBeta=0.1):
        
        weightVar=self.cvae.loss_weights['decoder_for_kl']
        K.set_value(weightVar,newBeta)
    
    def printWeights(self,mondule_names=['encoder']):
        if('encoder' in mondule_names):
            input_names=self.encoder.input_names
            for layer in self.encoder.layers:
                if(layer.name not in input_names):
                    layer.trainable = True
                    
        if('decoder' in mondule_names):
            input_names=self.decoder.input_names
            for layer in self.decoder.layers:
                if(layer.name not in input_names):
                    layer.trainable = True
                
        if('embedding_enc' in mondule_names):
            input_names=self.embedding_enc.input_names
            for layer in self.embedding_enc.layers:
                if(layer.name not in input_names):
                    layer.trainable = True
        
        if('embedding_dec' in mondule_names):
            input_names=self.embedding_dec.input_names
            for layer in self.embedding_dec.layers:
                if(layer.name not in input_names):
                    layer.trainable = True


    
