import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.FakeNewsDataModule import FakeNewsDataModule
from modules.AttentionLSTMClf import AttentionLSTMClf


def setup():
    with open("../config.json") as f:
        config = json.load(f)
    data_module = FakeNewsDataModule(bodies_path=config['bodies_path'],
                                     stances_path=config['stances_path'],
                                     w2v_model_path=config['w2v_model_path'],
                                     batch_size=config['batch_size'],
                                     train_frac=config['train_frac'],
                                     test_frac=config['test_frac'],
                                     num_workers=config['num_workers'],
                                     text_input_len=config['text_input_len'])
    model = AttentionLSTMClf(n_features=config['n_features'],
                             hidden_size=config['hidden_size'],
                             num_layers=config['num_layers'],
                             dropout=config['dropout'],
                             sequence_length=config['sequence_length'],
                             out_features=config['out_features'],
                             learning_rate=config['learning_rate'],
                             batch_size=config['batch_size'])
    checkpoint_callback = ModelCheckpoint(
        monitor=config['model_checkpoint_monitor'],
        dirpath=config['ckpt_save_path'],
        filename=config['checkpoint_save_filename'],
    )

    trainer = pl.Trainer(default_root_dir=config['ckpt_save_path'],
                         max_epochs=config['max_epochs'],
                         gpus=config['num_gpus'],
                         callbacks=[checkpoint_callback])

    return data_module, model, trainer
