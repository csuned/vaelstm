import torch as pt
from torch import nn
from torch.nn import functional as F
import pdb


class BetaVAE(nn.Module):
    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 batchsize: int,
                 total_batch: int,
                 hidden_dims = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e4,
                 loss_type: str = 'B',
                 seq_len: int = 48,
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = pt.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.M_N = batchsize/total_batch

        modules = []
        if hidden_dims is None:
            #hidden_dims = [in_channels, in_channels, in_channels]
            hidden_dims = [32, 64, 128, 256]
        kernel_size = (3,1)
        stride=(2,1)
        # Build Encoder
        for idx, h_dim in enumerate(hidden_dims):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=kernel_size, stride=stride, padding =  ((kernel_size[0]-2),0)),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        assert seq_len%(2**len(hidden_dims))==0, 'The Sequence length should be n*2^encode_layers.'
        self.hidden_len = int(seq_len/(2**len(hidden_dims)))
        latent_dim = self.hidden_len*hidden_dims[-1]
        self.fc_mu = nn.Linear(hidden_dims[-1]*self.hidden_len, latent_dim) # hidden_dims[-1]*4
        self.fc_var = nn.Linear(hidden_dims[-1]*self.hidden_len, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] *self.hidden_len) # hidden_dims[-1]*4

        hidden_dims.reverse()
#         for i in range(len(hidden_dims) - 1):
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(hidden_dims[i],
#                               hidden_dims[i + 1], #equal to filter in tf
#                               kernel_size=kernel_size,
#                               stride = 1,
#                               padding='same'),
#                     nn.BatchNorm2d(hidden_dims[i + 1]),
#                     nn.LeakyReLU())
#             )
        
        #In general, VAE use transpose or pixelshuffle here, but according to the paper we should samply is conv2d
        #a tranpose version is enclosed FYI.
        
        decode_padding = ((kernel_size[0]-2),0)
        output_padding = (1,0)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=kernel_size,
                                       stride = stride,
                                       padding=decode_padding,
                                       output_padding=output_padding),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)
        
#         self.final_layer = nn.Sequential(
#                             nn.Conv2d(hidden_dims[-1],
#                                       hidden_dims[-1],
#                                       kernel_size=kernel_size,
#                                       stride=1,
#                                       padding='same'),
#                             nn.BatchNorm2d(hidden_dims[-1]),
#                             nn.LeakyReLU(),
#                             nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels,
#                                       kernel_size= kernel_size, padding='same'),
#                             nn.Tanh())

#         #the paper be Using Conv2D
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=decode_padding,
                                               output_padding=output_padding),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels,
                                      kernel_size=kernel_size, padding='same'),
                            nn.Tanh())

    def encode(self, input_):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input_) #[32, 256, 9, 1] = [batchsize, channel, seqlen/16, seqdim]
        result = pt.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result) #[32, 256, 9, 1] -> [32, 256x9]
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z) #[32, 256x9],[32, 256x9] -> [32, 256x9]
        result = result.view(-1, 256, self.hidden_len, 1) #[32, 256x9] -> [32, 256, 9, 1]
        result = self.decoder(result) #[32, 256, 9, 1] -> [32, 2, 144, 1]
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = pt.exp(0.5 * logvar)
        eps = pt.randn_like(std)
        return eps * std + mu

    def forward(self, input_):
        mu, log_var = self.encode(input_)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input_, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input_ = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = self.M_N  # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input_)

        kld_loss = pt.mean(-0.5 * pt.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input_.device)
            C = pt.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples,
               current_device,):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = pt.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """