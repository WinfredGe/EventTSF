import sys
from lightning.pytorch.callbacks import TQDMProgressBar
from tqdm import tqdm as Tqdm


class MyTQDMProgressBar(TQDMProgressBar):

    def __init__(self):
        super(MyTQDMProgressBar, self).__init__()

    def init_validation_tqdm(self):

        bar = Tqdm(
            desc=self.validation_description,
            position=0,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_test_tqdm(self):

        bar = Tqdm(
            desc=self.test_description,
            position=0,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_predict_tqdm(self):

        bar = Tqdm(
            desc=self.predict_description,
            position=0,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar