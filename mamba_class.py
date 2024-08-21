import sys
sys.path.append('/home/mariapap/CODE/try/Class_Location_v2/mamba_train/')

import itertools
import argparse
import time
from mamba_train.config import get_config
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from mamba_train.changedetection.models.STMambaBDA import STMambaBDA
import os
import numpy as np

class Trainer(object):
    def __init__(self, format, mweights, args):
        self.args = args
        config = get_config(args)

        # self.train_data_loader = make_data_loader(args)


        self.deep_model = STMambaBDA(
            output_building=2, output_damage=5,
            pretrained=mweights, #args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE,
            in_chans=config.MODEL.VSSM.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VSSM.DEPTHS,
            dims=config.MODEL.VSSM.EMBED_DIM,
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK ==
                         "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        )
        self.deep_model = self.deep_model.cuda()
#        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
#                                            args.model_type + '_' + str(time.time()))
        # self.lr = args.learning_rate
        # self.epoch = args.max_iters // args.batch_size

        # if not os.path.exists(self.model_save_path):
        #    os.makedirs(self.model_save_path)

        if mweights is not None:
            if not os.path.isfile(mweights):
                raise RuntimeError(
                    "=> no checkpoint found at '{}'".format(mweights))
            checkpoint = torch.load(mweights)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.deep_model.eval()

    def normalize_img(self, img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
        """Normalize image by subtracting mean and dividing by std."""
        img_array = np.asarray(img)
        normalized_img = np.empty_like(img_array, np.float32)

        for i in range(3):  # Loop over color channels
            normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]

        return normalized_img

    def validation(self, cfile):
        #print('---------starting evaluation-----------')
#        self.evaluator_loc.reset()
        # self.evaluator_clf.reset()
        # dataset = DamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        # val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
            torch.cuda.empty_cache()

        # vbar = tqdm(val_data_loader, ncols=50)
#        ids = os.listdir(self.args.dataset_path)
#        for itera, id in enumerate(tqdm(ids)):

#            pre_change_imgs = Image.open(cfile)
            # labels_loc = Image.open(self.args.dataset_path + '{}'.format(id))

#            pre_change_imgs = np.array(pre_change_imgs)
#            print(cfile)
            pre_change_imgs = self.normalize_img(cfile)

            # idx255 = np.where(labels_loc==255)
            # labels_loc[idx255]=1

            # print('shapeeeeweeeeeeeeee', pre_change_imgs.shape, labels_loc.shape)

            pre_change_imgs = np.transpose(pre_change_imgs, (2, 0, 1))
            pre_change_imgs = torch.from_numpy(pre_change_imgs).unsqueeze(0)
            # labels_loc = torch.from_numpy(labels_loc)

            pre_change_imgs = pre_change_imgs.cuda()
            # post_change_imgs = post_change_imgs.cuda()
            # labels_loc = labels_loc.cuda().long()
            # print(np.unique(labels_loc.data.cpu().numpy()))

            # labels_clf = labels_clf.cuda().long()

            # input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
            with torch.no_grad():
#                output = self.predict(pre_change_imgs, self.deep_model, step=32, patch_size=(128, 128))

#                output = (self.predict(pre_change_imgs, self.deep_model, step=32, patch_size=(128, 128)))
#                output = np.argmax(output, 2)
                output_loc = self.deep_model(pre_change_imgs)
#            output_loc = torch.argmax(output_loc, 1).squeeze()
#            print('shapeeee', output_loc.shape)

            output_loc = output_loc.data.cpu().numpy()
            output_loc = np.argmax(output_loc, axis=1).squeeze()

            # labels_loc = labels_loc.cpu().numpy()

#            print('output', output_loc.shape)
#            print('uni', np.unique(output_loc))
#            print('uni', np.unique(output_loc))
            return output_loc
#            cv2.imwrite('./PREDS/before/{}.png'.format(id[:-4]), output_loc*255)

            # output_clf = output_clf.data.cpu().numpy()
            # output_clf = np.argmax(output_clf, axis=1)
            # labels_clf = labels_clf.cpu().numpy()

            # self.evaluator_loc.add_batch(labels_loc, output_loc)

            # output_clf = output_clf[labels_loc > 0]
            # labels_clf = labels_clf[labels_loc > 0]
            # self.evaluator_clf.add_batch(labels_clf, output_clf)

#        loc_recall_score = self.evaluator_loc.Pixel_Precision_Rate()
#        loc_prec_score = self.evaluator_loc.Pixel_Recall_Rate()
#        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        # damage_f1_score = self.evaluator_clf.Damage_F1_socore()
        # harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        # oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1
#        print(f'F1 is {loc_f1_score}, Recall is {loc_recall_score}, Precision is {loc_prec_score}')
