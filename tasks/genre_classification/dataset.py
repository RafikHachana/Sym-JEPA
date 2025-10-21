

class GenreClassificationDataset(MidiDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        super().setup(stage)
