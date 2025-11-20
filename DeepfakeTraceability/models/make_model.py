from . import MODEL_FACTORY
def make_model(config, num_classes=10):
<<<<<<< HEAD
    model = MODEL_FACTORY[config['model']['name']](config, num_classes=num_classes)
=======
    model = MODEL_FACTORY[config['name']](config, num_classes=num_classes)
>>>>>>> d44a48f6e3377ca4ca378484fc4c034268057bb8
    return model
