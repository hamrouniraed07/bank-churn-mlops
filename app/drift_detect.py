"""
Module de détection de drift pour le monitoring du modèle
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def detect_drift(
    reference_file: str,
    production_file: str,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Détecte le drift entre les données de référence et de production
    en utilisant le test de Kolmogorov-Smirnov
    
    Args:
        reference_file: Chemin vers les données de référence
        production_file: Chemin vers les données de production
        threshold: Seuil de p-value pour détecter le drift
        
    Returns:
        Dictionnaire avec les résultats du drift par feature
    """
    try:
        # Charger les données
        df_ref = pd.read_csv(reference_file)
        df_prod = pd.read_csv(production_file)
        
        results = {}
        
        # Features numériques à analyser
        numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                          'NumOfProducts', 'EstimatedSalary']
        
        for feature in numeric_features:
            if feature in df_ref.columns and feature in df_prod.columns:
                # Test de Kolmogorov-Smirnov
                statistic, p_value = stats.ks_2samp(
                    df_ref[feature].dropna(),
                    df_prod[feature].dropna()
                )
                
                results[feature] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "drift_detected": p_value < threshold
                }
        
        return results
        
    except FileNotFoundError as e:
        logger.warning(f"Fichier de données non trouvé: {e}")
        # Retourner un résultat vide plutôt qu'une erreur
        return {
            "error": "Fichiers de données non disponibles",
            "message": "La détection de drift nécessite des données de production"
        }
    except Exception as e:
        logger.error(f"Erreur lors de la détection de drift: {e}")
        raise
