
'''PaviaU###########'''
config_paviau = {}
config_paviau['dataset'] = 'paviau'
config_paviau['verbose'] = True
config_paviau['save_every_epoch'] = 50
config_paviau['print_every'] = 100

config_paviau['lr'] = 1e-3
config_paviau['lr_schedule'] = 'manual'  # manual, plateau, or a number
config_paviau['batch_size'] = 61
config_paviau['epoch_num'] = 150
config_paviau['epoch_num2'] = 100
config_paviau['init_std'] = 0.0099999
config_paviau['init_bias'] = 0.0
config_paviau['batch_norm'] = True
config_paviau['batch_norm_eps'] = 1e-05
config_paviau['batch_norm_decay'] = 0.9
config_paviau['conv_filters_dim'] = 3

config_paviau['e_pretrain'] = False
config_paviau['e_pretrain_sample_size'] = 63

config_paviau['e_num_filters'] = 64
config_paviau['e_num_layers'] = 3

config_paviau['g_num_filters'] = 64
config_paviau['g_num_layers'] = 3

config_paviau['zdim'] = 64
config_paviau['cost'] = 'l2sq'  # l2, l2sq, l1
config_paviau['lambda'] = 0.1
config_paviau['n_classes'] = 9

config_paviau['mlp_classifier'] = False
# config_paviau['eval_strategy'] = 1
config_paviau['sampling_size'] = 10
config_paviau['augment_z'] = True
config_paviau['augment_x'] = False
config_paviau['aug_rate'] = 1
config_paviau['LVO'] = True
config_paviau['window_size'] = 13
config_paviau['num_pcs'] =103
config_paviau['dataset_ratio_mapping'] = [66, 186, 20, 30, 13, 50, 13, 36, 9]