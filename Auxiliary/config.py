from constants import get_vocab_path
from Auxiliary.utils import printConfig

class Configuration(object):

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if hasattr(self, 'dataset'):
            self.vocab_path = get_vocab_path(self.dataset)



