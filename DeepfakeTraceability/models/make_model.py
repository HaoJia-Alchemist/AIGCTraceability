from . import MODEL_FACTORY
def make_model(config, num_classes=10):
    model = MODEL_FACTORY[config['model']['name']](config, num_classes=num_classes)
    return model
