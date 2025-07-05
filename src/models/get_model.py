from src.models.model.chembert import ChemBERT

# import ChemBERTa model
def get_model(args):

    model = ChemBERT(
        model_name=args.model_name,
        mol_f_dim=args.mol_f_dim,
        out_dim=1
        )
    
    return model

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from configs.training_arguements import get_arguments

    args = get_arguments()
    model = get_model(args)
    breakpoint() # debug