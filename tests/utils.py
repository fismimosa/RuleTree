def load_dataset(file_name):
    """
    Carica il dataset dal file specificato.
    
    Parameters:
        file_name (str): Percorso del file del dataset
        
    Returns:
        tuple: (X, y) dove X è la matrice delle features e y è il vettore target
    """
    import pandas as pd
    
    # Carica il file CSV
    df = pd.read_csv(file_name)
    
    # Assume che l'ultima colonna sia il target
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    
    return X.values, y.values