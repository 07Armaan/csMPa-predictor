from sklearn.base import BaseEstimator,TransformerMixin

class outliers_handling(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.bounds = {}

    def fit(self,x,y=None):
        numeric_cols = x.select_dtypes(include=["int64","float64"]).columns
        for col in numeric_cols:
            q1 = x[col].quantile(0.25)
            q3 = x[col].quantile(0.75)
            iqr = q3-q1
            lower = q1-1.5*iqr
            upper = q3+1.5*iqr
            self.bounds[col] = (lower,upper)
        return self
    
    def transform(self,x):
        x = x.copy()
        numeric_cols = x.select_dtypes(include=["int64","float64"]).columns
        for col in numeric_cols:
            if col in self.bounds:
                lower,upper = self.bounds[col]
                x[col] = x[col].clip(lower,upper)
        return x
    
    def get_feature_names_out(self,input_features=None):
        return input_features