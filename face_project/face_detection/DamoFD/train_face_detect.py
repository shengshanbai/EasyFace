import os
import tempfile
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.hub.snapshot_download import snapshot_download

model_id = 'iic/cv_ddsar_face-detection_iclr23-damofd-34G'
# remove '_mini' for full dataset
ms_ds_widerface = MsDataset.load('WIDER_FACE', namespace='shaoxuan')

data_path = ms_ds_widerface.config_kwargs['split_config']
train_dir = data_path['train']
val_dir = data_path['validation']


def get_name(dir_name):
    names = [i for i in os.listdir(dir_name) if not i.startswith('_')]
    return names[0]


train_root = train_dir + '/' + get_name(train_dir) + '/'
val_root = val_dir + '/' + get_name(val_dir) + '/'
cache_path = snapshot_download(model_id)
tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)


def _cfg_modify_fn(cfg):
    cfg.checkpoint_config.interval = 1
    cfg.log_config.interval = 10
    cfg.evaluation.interval = 1
    cfg.data.workers_per_gpu = 1
    cfg.data.samples_per_gpu = 4
    return cfg


kwargs = dict(
    cfg_file=os.path.join(cache_path, 'DamoFD_lms.py'),
    work_dir=tmp_dir,
    train_root=train_root,
    val_root=val_root,
    total_epochs=1,  # run #epochs
    cfg_modify_fn=_cfg_modify_fn)

trainer = build_trainer(
    name=Trainers.face_detection_scrfd, default_args=kwargs)
trainer.train()
