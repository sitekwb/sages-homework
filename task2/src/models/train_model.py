import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")


from setup import setup


def train_model():
    data_module, model, trainer = setup()

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    train_model()
