from models.ast_flat import AstFlatGRUModel as astflat
from models.ast_flat_fullemb import AstFlatFullEmbGRUModel as astflatfullemb
from models.ast_gnn import AstGnnGRUModel as astgnn
from models.ast_gnn_fullemb import AstGNNFullEmbGRUModel as astgnnfullemb
def create_model(modeltype, config):
    mdl = None

    if modeltype == 'astflat':
    	# predict first word based on flat AST only
        mdl = astflat(config)
    elif modeltype == 'astflat-fullemb':
    	# predict first word based on flat AST only
        mdl = astflatfullemb(config)
    elif modeltype == 'astgnn':
    	# predict first word based on flat AST nodes and edges
        mdl = astgnn(config)
    elif modeltype == 'astgnn-fullemb':
        mdl = astgnnfullemb(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
