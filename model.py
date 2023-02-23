import tensorflow.keras as keras
import tensorflow as tf

from models.attendgru import AttentionGRUModel as attendgru
from models.ast_attendgru import AstAttentionGRUModel as ast_attendgru
from models.attendgru_fc import AttentionGRUFCModel as attendgru_fc
from models.ast_attendgru_fc import AstAttentionGRUFCModel as ast_attendgru_fc
from models.codegnngru import CodeGNNGRUModel as codegnngru
from models.code2seq import Code2SeqModel as code2seq
from models.cmc_cg0 import CmcCG0Model as cmc_cg0
from models.cmc_cg1 import CmcCG1Model as cmc_cg1
from models.cmc_cg2 import CmcCG2Model as cmc_cg2
from models.cmc_cg3 import CmcCG3Model as cmc_cg3
from models.qstransformer import QSTransformer as qs_xformer

from models.attendgru_bio import AttentionGRUBioModel as attendgru_bio
from models.attendgru_bio_base import AttentionGRUBioBaseModel as attendgru_bio_base
from models.attendgru_bio_base2 import AttentionGRUBioBase2Model as attendgru_bio_base2

from models.attendgru_bio2 import AttentionGRUBio2Model as attendgru_bio2

def create_model(modeltype, config):
    mdl = None

    if modeltype == 'attendgru':
        mdl = attendgru(config)
    elif modeltype == 'ast-attendgru': 
        mdl = ast_attendgru(config)
    elif modeltype == 'attendgru-fc':
        mdl = attendgru_fc(config)
    elif modeltype == 'codegnngru':
        mdl = codegnngru(config)
    elif modeltype == 'ast-attendgru-fc':
        mdl = ast_attendgru_fc(config)
    elif modeltype == 'code2seq':
        mdl = code2seq(config)
    elif modeltype == 'cmc-cg0':
        mdl = cmc_cg0(config)
    elif modeltype == 'cmc-cg1':
        mdl = cmc_cg1(config)
    elif modeltype == 'cmc-cg2':
        mdl = cmc_cg2(config)
    elif modeltype == 'cmc-cg3':
        mdl = cmc_cg3(config)
    elif modeltype == 'transformer':
        mdl = qs_xformer(config)
    elif modeltype == 'attendgru-bio':
        mdl = attendgru_bio(config)
    elif modeltype == 'attendgru-bio-base':
        mdl = attendgru_bio_base(config)
    elif modeltype == 'attendgru-bio-base2':
        mdl = attendgru_bio_base2(config)
    elif modeltype == 'attendgru-bio2':
        mdl = attendgru_bio2(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
