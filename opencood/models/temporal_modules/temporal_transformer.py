import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Union, List
from einops import rearrange, repeat
import math
from torch import optim
from opencood.tools.evaluation_temporal.utils import plot_embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    def __init__(self, feature_size=1*32*32, num_layers=6, num_heads=4, dropout=0.25, compress_only=False, pass_temporal=False, res_connection=False, **kwargs):
        super(TemporalTransformer, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size, dropout)
        
        encoder_layers = TransformerEncoderLayer(feature_size, num_heads, feature_size*4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(feature_size, feature_size)

        self.small_channel_number = 1
        self.original_channel_number = 128

        self.compressor_decompressor = Compressor_Decompressor(self.small_channel_number, self.original_channel_number)

        self.dummy_cnn = nn.Conv2d(self.original_channel_number, self.original_channel_number, kernel_size=3, padding=1, stride=1)

        # kompress and pass can not be both true but both
        assert not (compress_only and pass_temporal)
        assert not (pass_temporal and res_connection)
        self.kompress_only = compress_only
        self.pass_temporal = pass_temporal
        self.res_connection = res_connection

        self.fix_compressor = False
        if self.fix_compressor:
            for param in self.compressor_decompressor.parameters():
                param.requires_grad = False
        
        self.sequence_len = 4
    
    def forward(self, current_embedding, past_embeddings, **kwargs):
        B, C, H, W = current_embedding.shape
        past_embeddings = past_embeddings.permute(1,0,2,3,4)
        num_histories = past_embeddings.shape[0]
        # ignore all sequence except the last one
        if num_histories == self.sequence_len - 1:
            if self.pass_temporal:
                return current_embedding
            src = self._prepare_sequence(current_embedding, past_embeddings)
            if self.kompress_only:
                output = src[-1]
            else:
                output = self.transformer_encoder(src, self.src_mask)
                output = self.decoder(output)[-1]
            output = rearrange(output, 'b (c h w) -> b c h w', c=1, h=H, w=W)
            output = self._prepare_output(output)
            return output


        else:
            return current_embedding

    def _prepare_sequence(self, x: torch.Tensor, sequence_list: torch.Tensor) -> torch.Tensor:
        """
        Prepares the sequence for the transformer encoder.
        :param x: the current embedding. Shape: (B, C, H, W)
        :param sequence_list: the past embeddings. Shape: (B, S, C, H, W)
        """
        # append the sequence elements to the x tensor
        x = torch.cat([sequence_list, x.unsqueeze(0)], dim=0)

        x = self.compressor_decompressor.compress(x)

        s, b, c, h, w = x.shape
        x = x.view(s,b,c*h*w)
        return x
    
    def _sequence_to_batch(self, sequence: List[torch.Tensor]) -> torch.Tensor:
        s,b,c,h,w = sequence.shape
        sequence = sequence.view(s*b,c,h,w)
        return sequence
    
    def _batch_to_sequence(self, batched_sequence: torch.Tensor) -> List[torch.Tensor]:
        b,c,h,w = batched_sequence.shape
        sequence = batched_sequence.view(self.sequence_len, int(b / (self.sequence_len)), c, h, w)
        return sequence
    

    def _prepare_output(self, x: torch.Tensor) -> torch.Tensor:
        #x = x.view(1,self.small_channel_number,32,32)
        x = self.compressor_decompressor.decompress(x)
        return x

    def _change_num_channels(self, x: torch.Tensor, num_init_channels, num_target_channels) -> torch.Tensor:
        if num_init_channels > num_target_channels:
            conv = self.channel_reducer
        elif num_init_channels < num_target_channels:
            conv = self.channel_increaser
        else:
            return x
        if len(x.shape) == 5:
            s, b, c, h, w = x.shape
            x = x.view(s*b, c, h, w)
            x = conv(x)
            x = x.view(s, b, num_target_channels, h, w)
        elif len(x.shape) == 4:
            x = conv(x)
        return x

    def _full_sequence(self, sequence_list: List[torch.Tensor]):
        if len(sequence_list) >= self.sequence_len - 1:
            return True
        else:
            return False

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class Compressor_Decompressor(nn.Module):
    def __init__(self, num_target_channels=1, num_init_channels=128):
        super(Compressor_Decompressor, self).__init__()
        self.num_target_channels = num_target_channels
        self.num_init_channels = num_init_channels
        
        self.kompressor_bn = nn.Sequential(
                    nn.Conv2d(self.num_init_channels, 64, kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    
                    nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    
                    nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    
                    nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1),
                )

        self.dekompressor_bn = nn.Sequential(
                nn.ConvTranspose2d(1, 16, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                
                nn.ConvTranspose2d(16, 32, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                
                nn.ConvTranspose2d(32,64, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                nn.ConvTranspose2d(64, self.num_init_channels, kernel_size=3, padding=1, stride=1),
            )
        '''
        self.kompressor_bn = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            #nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            #nn.BatchNorm2d(128),
            #nn.ReLU(),

            #nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            #nn.BatchNorm2d(128),
            #nn.ReLU(),

            #nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
        )

        self.dekompressor_bn = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            #nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=1),
            #nn.BatchNorm2d(128),
            #nn.ReLU(),

            #nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=1),
            #nn.BatchNorm2d(128),
            #nn.ReLU(),

            #nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=1),
        )
        '''    
        

            
    
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        s, b, c, h, w = x.shape
        x = rearrange(x, 's b c h w -> (s b) c h w')
        x = self.kompressor_bn(x)
        x = rearrange(x, '(s b) c h w -> s b c h w', s=s)
        return x


    def decompress(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dekompressor_bn(x)
        return x


# test the model
if __name__ == "__main__":
    sequence_len = 4
    kwargs = {'sequence_len': sequence_len}
    tensor_size = (1,128,32,32)
    # init the model
    print(f'creating model')
    model = TemporalTransformer(**kwargs)

    # create random input tensor with size of tensor size
    x = torch.rand(tensor_size)

    # create random sequence_list
    sequence_list = [torch.rand(tensor_size) for _ in range(sequence_len)]

    # run the model
    output = model(x, sequence_list)
    print(output.shape)
