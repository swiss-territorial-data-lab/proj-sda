
class UnknownScoreType(Exception):

    def __init__(self, threshold_score_type):
        super().__init__(threshold_score_type)

    def __str__(self):
        return f'Unknown type of score threshold: {self.threshold_score_type}. Should be conservative or optimal.'
    
    pass

DONE_MSG = "...done."
SCATTER_PLOT_MODE = 'markers+lines'
OVERWRITE = True
KEEP_DATASET_SPLIT = True
THRESHOLD_PER_MODEL = {68: 0.5, 72: 0.4, 75: 0.2, 92: 0.25, 102: 0.35}
SCORE_THRESHOLD_TYPE = 'optimal'    #  supported values: 1. conservative (= 0.05), 2. optimal (best for the model f1-score as defined in THRESHOLD_PER_MODEL)