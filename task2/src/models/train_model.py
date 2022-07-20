from setup import setup


def train_model():
    data_module, model, trainer = setup()

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == '__main__':
    train_model()
