import pandas as pd
import time
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, precision_score

def load_data(file_path):
    with open(file_path, 'r') as file:
        df = pd.read_csv(file)
    return df

def preprocess_data(df):
    assert not df.isnull().any().any(), "В датасете есть пропущенные значения."
    assert not df.duplicated().any(), "В датасете есть дубликаты."
    print("Предобработка данных завершена.")
    
    ranked_sessions = df.groupby('query_id', as_index=False, group_keys=False).apply(lambda x: x.sort_values(by='rank', ascending=True), include_groups=False).reset_index(drop=True)
    
    features = ranked_sessions.loc[:, 'feature_0':'feature_143']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    target_variable = ranked_sessions['rank']
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target_variable, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Время выполнения функции '{func.__name__}': {round(end_time - start_time, 3)} секунд")
        return result
    return wrapper

@timer
def xgboost_regressor(X_train, X_test, y_train, y_test):
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)

    xgb_model.fit(X_train, y_train)

    xgb_y_pred = xgb_model.predict(X_test)

    xgb_ndcg = ndcg_score(y_test.values.reshape(1, -1), xgb_y_pred.reshape(1, -1), k=5)
    print("NDCG@5:", xgb_ndcg)

    precision = precision_score(y_test, xgb_y_pred > 0.07, average='macro', zero_division=1)
    print("Precision:", precision)

if __name__ == "__main__":
    file_path = 'intern_task.csv'
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    xgboost_regressor(X_train, X_test, y_train, y_test)
