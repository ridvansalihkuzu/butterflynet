"""
This code is generated by Ridvan Salih KUZU @DLR
LAST EDITED:  01.06.2021
ABOUT SCRIPT:
It contains some custom metric functions exploited for experimental purposes in this project.
"""
from keras_unet_collection import models
from model_zoo.mobile_unet import MobileUnet
from model_zoo.sim_clr import SimCLR
from model_zoo.classical_unet import Unet_A as Unet
from model_zoo.capsule_net_family.capsule_network import CapsNetR3
from model_zoo.double_unet import DoubleUnet
from model_zoo.adaptive_net_family.adaptive_attention_butterfly import AdaptiveAttentionButterfly


def get_sim_clr_model(base_model, input_shape):
    '''
      THIS FUNCTION RETURNS SIM-CLR MODEL.
      :param base_model: backbone UNET model
      :param input_shape: (width,height,channel) shape of the input images
      :return: returns simCLR model
     '''
    return SimCLR(base_model,input_shape)

def get_model(model_type, input_shape, is_freeze=False):
    '''
    THIS FUNCTION RETURNS a UNET MODEL VARIANT.
    :param model_type: model architecture index; 0: MobileUNet, 1: UNet, 2: ResUNet, 3:AttentionUNet with ResNet, 4:AttentionUNet with EfficientNet-B0, 5:AttentionUNet with EfficientNet-B7, 6:Attention UNet without backbone
    :param input_shape: (width,height,channel) shape of the input images
    :param is_freeze: it determine if the backbone will be frozen or not in training
    :return: returns a UNET model variant
    '''


    if model_type == 0:
        return MobileUnet(input_shape=input_shape, freeze_backbone= is_freeze)
    elif model_type == 1:
        return models.unet_2d(input_shape,[32, 64, 128, 256, 512],1,stack_num_down=5, stack_num_up=5,
                              batch_norm=True,pool=True,unpool=True,output_activation='Sigmoid',weights=None,freeze_batch_norm=False) #Unet(input_shape=input_shape)
    elif model_type == 2:
        return models.vnet_2d(input_shape, filter_num=[16, 32, 64, 128, 256], n_labels=1,
                      res_num_ini=1, res_num_max=3,activation='ReLU', output_activation='Sigmoid',batch_norm=True, pool=True, unpool=True, name='vnet')

    elif model_type == 3:
        return models.att_unet_2d(input_shape, [32, 64, 128, 256, 512], n_labels=1, stack_num_down=2, stack_num_up=2,
                                  backbone=None, activation='ReLU',
                                  atten_activation='ReLU', attention='multiply', output_activation='Sigmoid', batch_norm=True,
                                  pool=True,unpool=True, freeze_batch_norm=False, freeze_backbone=is_freeze, name='attunet')

    elif model_type == 4:
        return models.unet_plus_2d(input_shape, [32, 64, 128, 256, 512], n_labels=1,
                            stack_num_down=2, stack_num_up=2,
                            activation='ReLU', output_activation='Sigmoid',
                            batch_norm=False, pool=True, unpool=True, deep_supervision=True, name='xnet')

    elif model_type == 5:
        return models.unet_3plus_2d(input_shape, n_labels=1, filter_num_down=[32, 64, 128, 256, 512],
                             filter_num_skip='auto', filter_num_aggregate='auto',
                             stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Sigmoid',
                             batch_norm=True, pool=True, unpool=True, deep_supervision=True, name='unet3plus')

    elif model_type == 6:
        return models.r2_unet_2d(input_shape, [32, 64, 128, 256, 512], n_labels=1,
                          stack_num_down=2, stack_num_up=1, recur_num=2,
                          activation='ReLU', output_activation='Sigmoid',
                          batch_norm=True, pool='max', unpool=True, name='r2unet')

    elif model_type == 7:
        return models.resunet_a_2d(input_shape, [32, 64, 128, 256, 512],
                            dilation_num=[1, 3, 5, 7],
                            n_labels=1, aspp_num_down=256, aspp_num_up=128,
                            activation='ReLU', output_activation='Sigmoid',
                            batch_norm=True, pool=True, unpool=True, name='resunet')

    elif model_type == 8:
        return models.u2net_2d(input_shape, n_labels=1,
                        filter_num_down=[32, 64, 128, 256, 512],
                        activation='ReLU', output_activation='Sigmoid',
                        batch_norm=True, pool=True, unpool=True, deep_supervision=True, name='u2net')

    elif model_type == 9:
        return models.transunet_2d(input_shape, filter_num=[32, 64, 128, 256,512], n_labels=1, stack_num_down=2, stack_num_up=2,
                                embed_dim=512, num_mlp=1024, num_heads=12, num_transformer=12,
                                activation='ReLU', mlp_activation='GELU', output_activation='Sigmoid',
                                batch_norm=True, pool=True, unpool=True, name='transunet')

    elif model_type == 10:
        return models.swin_unet_2d(input_shape, filter_num_begin=32, n_labels=1, depth=4, stack_num_down=2, stack_num_up=2,
                            patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512,
                            output_activation='Sigmoid', shift_window=True, name='swin_unet')

    #elif model_type == 11:
    #    return CapsNetR3(input_shape=input_shape)
    #elif model_type == 11:
    #    return AdaptiveUNet(input_shape)
    elif model_type == 11:
        return DoubleUnet(input_shape)
    elif model_type == 12:
        return AdaptiveAttentionButterfly(input_shape, starting_filters=16)
    elif model_type == 13:
        return AdaptiveAttentionButterfly(input_shape, starting_filters=32)
    elif model_type == 14:
        return AdaptiveAttentionButterfly(input_shape, max_pools=3, starting_filters=16)
    elif model_type == 15:
        return AdaptiveAttentionButterfly(input_shape, max_pools=3,starting_filters=24)
    elif model_type == 16:
        return AdaptiveAttentionButterfly(input_shape, max_pools=3, starting_filters=32)
    elif model_type == 17:
        return AdaptiveAttentionButterfly(input_shape, max_pools=4,starting_filters=16)
    elif model_type == 19:
        return AdaptiveAttentionButterfly(input_shape, max_pools=4, starting_filters=24)
    elif model_type == 18:
        return AdaptiveAttentionButterfly(input_shape, max_pools=4,starting_filters=32)
    elif model_type == 20:
        return AdaptiveAttentionButterfly(input_shape, max_pools=5, starting_filters=16)
    elif model_type == 21:
        return AdaptiveAttentionButterfly(input_shape, max_pools=5,starting_filters=24)
    elif model_type == 22:
        return AdaptiveAttentionButterfly(input_shape, max_pools=5,starting_filters=32)






