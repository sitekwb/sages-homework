from setup import setup


def predict_model():
    data_module, model, trainer = setup()
    return trainer.predict(model, datamodule=data_module)
