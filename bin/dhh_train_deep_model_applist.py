# -*- coding: utf-8 -*-
import sys
sys.path.append("/search/odin/user/mingliang/model/dhh/dnn/lib/")

import numpy as np
import os
import common
import deep_model_applist as deep_model

#import logging
#logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s', level=logging.INFO)

available_gpuid_list = common.get_available_gpuid()
if len(available_gpuid_list) == 0:
    print("no available gpuid")
    sys.exit(1)
n = str(available_gpuid_list[0])
os.environ['CUDA_VISIBLE_DEVICES'] = n
print 'gpu-id: ', os.environ['CUDA_VISIBLE_DEVICES']

num_epoch = 3
file_read_config_param = {'field_delim':' ',
                        'epoch':2,
                        'batch_size':512,
                        'buffer_size':2,
                        'hidden_units':[256,256,256],
                        'n_classes':3,
                        'decay_steps':500,
                        'learning_rate':0.001, }
print("config_param:\n{}".format(file_read_config_param))

train_files = "data/train_applist.dat"
test_files = "data/test_applist.dat"
clf_file = "data/clf_applist.dat"
model_dir = "model/deepmodel"+"_applist_"+str(file_read_config_param['batch_size'])+"_"+str(file_read_config_param['learning_rate'])
deepmodel = deep_model.DeepModel(train_files, test_files, clf_file, file_read_config_param, model_dir)

print("-----training-----")
#deepmodel.train()
print("-----evaluating-----")
#deepmodel.evaluate()
deepmodel.train_and_evaluate()

print("-----finish predicting-----")
predictions = deepmodel.predict()
i=0
for pred in predictions:
    print pred
    i += 1
    if i >= 10:
        break

