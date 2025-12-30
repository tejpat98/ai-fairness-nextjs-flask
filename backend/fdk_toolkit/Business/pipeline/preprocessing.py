import pandas as pd

def preprocessing(dataset_path):
    try:
        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        
        if len(columns) < 3:
            return {"Error" :"Dataset too small. Need at least 3 columns."}
        
        # Business auto-detection
        column_mappings, column_reasoning = detect_business_column_mappings(df, columns)
        
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in column_mappings or not column_mappings[m]]
        
        if missing_required:
            return {"Error":f"Auto-Detection Failed. Could not automatically detect: {missing_required}. Please ensure your dataset has clear column names."}
        
        # Count actual key features detected
        detected_key_features = len([m for m in column_mappings.values() if m is not None])
        
        required_mappings = ['group', 'y_true', 'y_pred']
        missing_required = [m for m in required_mappings if m not in column_mappings or not column_mappings[m]]
        if missing_required:
            return {"Error":f"Missing required mappings: {missing_required}"}
        
        # Create clean DataFrame with mapped columns
        df_mapped = pd.DataFrame()
        
        for standard_name, original_name in column_mappings.items():
            if original_name and original_name in df.columns:
                df_mapped[standard_name] = df[original_name].copy()
        
        # Convert data types to Python native
        for col in df_mapped.columns:
            if df_mapped[col].dtype == 'bool':
                df_mapped[col] = df_mapped[col].astype(int)
            elif pd.api.types.is_integer_dtype(df_mapped[col]):
                df_mapped[col] = df_mapped[col].astype(int)
            elif pd.api.types.is_float_dtype(df_mapped[col]):
                df_mapped[col] = df_mapped[col].astype(float)
        
        # Validate required columns
        missing_cols = [col for col in required_mappings if col not in df_mapped.columns]
        if missing_cols:
            return {"Error":f"After mapping, missing columns: {missing_cols}"}
        
        # Validate each column is a proper Series
        for col in required_mappings:
            if not isinstance(df_mapped[col], pd.Series):
                return {"Error":f"Column '{col}' is not a Series."}
        
        return df_mapped
    
    except Exception as e:
        print( f"Error: {str(e)}")
        return {"Error":f"Error reading dataset or creating mappings."}
    
def detect_business_column_mappings(df, columns):
    """Auto-detection for business datasets"""
    suggestions = {'group': None, 'y_true': None, 'y_pred': None, 'y_prob': None}
    reasoning = {}
    
    for col in columns:
        reasoning[col] = ""
    
    # Layer 1: Direct matching for standard column names
    for col in columns:
        col_lower = col.lower()
        if col_lower in ['group', 'segment', 'customer_segment', 'demographic', 'category', 'cohort']:
            suggestions['group'] = col
            reasoning[col] = "Direct match: customer segment/group column"
            continue
        elif col_lower in ['y_true', 'actual', 'true', 'outcome', 'target', 'label', 'ground_truth', 'conversion']:
            suggestions['y_true'] = col
            reasoning[col] = "Direct match: true business outcomes/target variable"
            continue
        elif col_lower in ['y_pred', 'predicted', 'prediction', 'estimate', 'model_output', 'forecast']:
            suggestions['y_pred'] = col
            reasoning[col] = "Direct match: business model predictions"
            continue
        elif col_lower in ['y_prob', 'probability', 'score', 'confidence', 'risk_score', 'propensity', 'clv_score']:
            suggestions['y_prob'] = col
            reasoning[col] = "Direct match: probability/confidence scores"
            continue

    # Layer 2: Business-specific keyword detection
    for col in columns:
        if col in [suggestions['group'], suggestions['y_true'], suggestions['y_pred'], suggestions['y_prob']]:
            continue
            
        col_data = df[col]
        unique_vals = col_data.unique()
        
        # GROUP: Business-specific segments
        if col_data.dtype == 'object' or (col_data.nunique() <= 20 and col_data.nunique() > 1):
            business_group_keywords = ['customer_type', 'segment', 'cohort', 'region', 'market', 
                                     'loyalty_tier', 'age_group', 'income_bracket', 'geographic',
                                     'product_category', 'service_tier', 'marketing_channel']
            if any(keyword in col.lower() for keyword in business_group_keywords):
                suggestions['group'] = col
                reasoning[col] = "Business domain: Customer segments for fairness analysis"
                continue
                
        # Y_TRUE: Business outcomes
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) <= 10:
            if set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and min(unique_vals) in [0,1] and max(unique_vals) in [0,1]):
                business_true_keywords = ['conversion', 'purchase', 'churn', 'retention', 'response',
                                        'approval', 'engagement', 'satisfaction', 'loyalty',
                                        'campaign_success', 'service_usage', 'renewal']
                if any(keyword in col.lower() for keyword in business_true_keywords):
                    suggestions['y_true'] = col
                    reasoning[col] = "Business domain: Customer outcomes (binary: 0/1)"
                    continue
                    
        # Y_PRED: Business predictions
        if col_data.dtype in ['int64', 'float64'] and len(unique_vals) <= 10:
            if (set(unique_vals).issubset({0, 1}) or (len(unique_vals) == 2 and min(unique_vals) in [0,1] and max(unique_vals) in [0,1])) and col != suggestions['y_true']:
                business_pred_keywords = ['predicted_conversion', 'churn_risk', 'response_score', 
                                        'retention_prediction', 'engagement_forecast', 'clv_prediction',
                                        'recommendation_score', 'personalization_score']
                if any(keyword in col.lower() for keyword in business_pred_keywords):
                    suggestions['y_pred'] = col
                    reasoning[col] = "Business domain: Business algorithm predictions (binary: 0/1)"
                    continue
                    
        # Y_PROB: Probability scores
        if col_data.dtype in ['float64', 'float32']:
            if len(unique_vals) > 2 and (col_data.between(0, 1).all() or (col_data.min() >= 0 and col_data.max() <= 1)):
                prob_keywords = ['probability', 'score', 'confidence', 'likelihood', 'propensity',
                               'estimate', 'calibration', 'confidence_score', 'rating', 'clv']
                if any(keyword in col.lower() for keyword in prob_keywords):
                    suggestions['y_prob'] = col
                    reasoning[col] = "Business domain: Business probability scores (0-1 range)"
                    continue
    
    # Layer 3: Statistical fallbacks
    if not suggestions['group']:
        for col in columns:
            if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 20:
                suggestions['group'] = col
                reasoning[col] = "Statistical fallback: Customer segments (2-20 unique values)"
                break
        if not suggestions['group']:
            for col in columns:
                if df[col].dtype in ['int64', 'float64'] and 2 <= df[col].nunique() <= 10:
                    suggestions['group'] = col
                    reasoning[col] = "Statistical fallback: Numeric segments (2-10 unique values)"
                    break
                
    if not suggestions['y_true']:
        for col in columns:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                if col != suggestions['y_pred']:
                    suggestions['y_true'] = col
                    reasoning[col] = "Statistical fallback: Binary outcomes (2 unique values)"
                    break
                
    if not suggestions['y_pred']:
        for col in columns:
            if (col != suggestions['y_true'] and df[col].dtype in ['int64', 'float64'] 
                and df[col].nunique() == 2):
                suggestions['y_pred'] = col
                reasoning[col] = "Statistical fallback: Binary predictions (2 unique values)"
                break
    
    return suggestions, reasoning
