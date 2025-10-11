from . import MODEL_FACTORY
def make_model(config, num_classes=10):
    model = MODEL_FACTORY[config['name']](config, num_classes=num_classes)
    return model
