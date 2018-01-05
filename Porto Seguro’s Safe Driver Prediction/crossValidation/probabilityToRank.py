import pandas as pd

def probability_to_rank(prediction, scaler=1):
    """
    convert probability into rank for these two OOF function.
    The needs to use normalised rank instead of predicted probabilities will
    become appearent later in this notebook
    """
    pred_df = pd.DataFrame(columns=['probability'])
    pred_df['probability'] = prediction
    pred_df['rank'] = pred_df['probability'].rank()/len(prediction)*scaler
    return pred_df['rank'].values
