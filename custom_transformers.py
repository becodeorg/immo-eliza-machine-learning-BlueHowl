from sklearn.base import BaseEstimator, TransformerMixin

from outlier_util import remove_outliers_iqr
from peb_util import map_label_to_kwh


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops specified columns from a DataFrame."""

    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        return x.drop(columns=self.columns_to_drop)
    
class NADropper(BaseEstimator, TransformerMixin):
    """Transformer that drops rows with NaN values in specified columns."""

    def __init__(self, subset=None):
        self.subset = subset

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        return x.dropna(subset=self.subset)
    
class NAReplacer(BaseEstimator, TransformerMixin):
    """Transformer that replaces specified values in a DataFrame."""

    def __init__(self, column, new_value):
        self.column = column
        self.new_value = new_value

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_copy = x.copy()
        
        if isinstance(self.column, list):
            # Handle list
            for col in self.column:
                new_val = self.new_value
                
                if self.new_value == 'median': 
                    new_val = int(x_copy[col].median())
                elif self.new_value == 'mode':
                    new_val = x_copy[col].mode()[0]

                x_copy[col] = x_copy[col].fillna(new_val)
        else:  
            new_val = self.new_value 
            
            if self.new_value == 'median':
                new_val = int(x_copy[self.column].median())
            elif self.new_value == 'mode':
                new_val = x_copy[self.column].mode()[0]

            x_copy[self.column] = x_copy[self.column].fillna(new_val)
        
        return x_copy

class DuplicateDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops duplicate rows from a DataFrame."""

    def __init__(self, subset=None):
        self.subset = subset
        pass

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        return x.drop_duplicates(self.subset)
    
class EpcKwhCalculator(BaseEstimator, TransformerMixin):
    """Transformer that calculates epc_kwh from epcScore & province."""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x_copy = x.copy()

        x_copy["epc_kwh"] = x_copy.apply(map_label_to_kwh, axis=1)

        return x_copy
    
class OutlierRemover(BaseEstimator, TransformerMixin):
    """Transformer that removes outliers from specified columns using the IQR method."""

    def __init__(self, columns=None, multiplier=1.5, min_thresholds=None):
        self.columns = columns
        self.multiplier = multiplier
        self.min_thresholds = min_thresholds if min_thresholds is not None else {}

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        return remove_outliers_iqr(x.copy(), self.columns, self.multiplier, self.min_thresholds)

class CategoryTransformer(BaseEstimator, TransformerMixin):
    """Transformer that converts specified columns to categorical type."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x_copy = x.copy()

        if self.columns is not None:
            for col in self.columns:
                order_by_price = x_copy.groupby(col)['price'].mean().round().sort_values(ascending=False)
                x_copy[col] = x_copy[col].map(order_by_price.to_dict())
        
        return x_copy
    
class PostalCodeTransformer(BaseEstimator, TransformerMixin):
    """Transformer that converts postalCode to an order of importance."""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x_copy = x.copy()

        x_copy['price_per_m2'] = x_copy['price'] / x_copy['habitableSurface']
        postal_code_price = x_copy.groupby('postCode')['price_per_m2'].mean().round().sort_values(ascending=False)
        x_copy['postCode'] = x_copy['postCode'].map(postal_code_price.to_dict())

        x_copy.drop(columns=['price_per_m2'], inplace=True)

        return x_copy
    
class BooleanTransformer(BaseEstimator, TransformerMixin):
    """Transformer that converts boolean strings to integers (True=1, False=0)."""
    
    def __init__(self):
        pass
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x_copy = x.copy()
        
        bool_columns = x_copy.select_dtypes(include='bool').columns
        for column in bool_columns:
            x_copy[column] = x_copy[column].astype(int)
            
        return x_copy
    
class ToIntTransformer(BaseEstimator, TransformerMixin):
    """Transformer that converts columns to int, removing decimal parts."""

    def __init__(self, for_display=False):
        self.for_display = for_display

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x_copy = x.copy()

        for col in x_copy.columns:
            x_copy[col] = x_copy[col].astype(int)
        
        return x_copy