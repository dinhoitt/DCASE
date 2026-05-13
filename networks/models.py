from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE
from networks.med_asd.med_asd import MEDASD

class Models:
    ModelsDic = {
        "DCASE2023T2-AE":DCASE2023T2AE,
        "MED-ASD":MEDASD,
    }

    def __init__(self,models_str):
        self.net = Models.ModelsDic[models_str]

    def show_list(self):
        return Models.ModelsDic.keys()
