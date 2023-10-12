class args():
    traindata_dir = '/your/train/data'
    testdata_dir = '/Data/testQB'
    evaldata_dir = '/your/eval/data'
    oritestdata_dir = '/your/test/data'
    sample_dir = './sample'
    checkpoint_dir = './checkpoint'
    checkpoint_backup_dir = './checkpoint/backup'
    record_dir = './log/record'
    log_dir = './log'
    
    model2_path = 'model/pre_train/fuse_v7.pth'
    output_dir = './output'
    edge_enhance_pretrain_model = './checkpoint/edge_enhance_trained.pth'
    edge_enhance_multi_pretrain_model = './checkpoint/edge_enhance_multi.pth'
    pretrain_model = ''

    max_value = 1023 
    epochs = 300
    lr = 0.0005  # learning rate
    batch_size_net = 16
    batch_size = 8
    lr_decay_freq = 40
    model_backup_freq = 30
    eval_freq = 10


    data_augmentation = False

    log_interval = 1000  # number of images after which the training loss is logged
    cuda = 1    # use GPU 1