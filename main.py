import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import random
from deap import creator, base, tools, algorithms

def preprocess_waterbase_data(file_path):
    print("Loading data...")
    data = pd.read_csv(file_path, low_memory=False)
    print(f"Raw data shape: {data.shape}")
    data = data.dropna(subset=['resultMeanValue'])
    print(f"After removing missing mean values: {data.shape}")
    data['LOQ'] = 'Good'
    data.loc[data['resultQualityNumberOfSamplesBelowLOQ'] == data['resultNumberOfSamples'], 'LOQ'] = 'Bad'
    data = data[data['LOQ'] == 'Good']
    print(f"After removing low-quality samples: {data.shape}")
    relevant_cols = [
        'monitoringSiteIdentifier', 
        'parameterWaterBodyCategory', 
        'observedPropertyDeterminandLabel', 
        'phenomenonTimeReferenceYear', 
        'resultMeanValue'
    ]
    data = data[relevant_cols]
    print(f"After keeping only relevant features: {data.shape}")
    data = data.drop_duplicates(subset=[
        'monitoringSiteIdentifier', 
        'observedPropertyDeterminandLabel', 
        'phenomenonTimeReferenceYear'
    ])
    print(f"After removing duplicates: {data.shape}")
    print("Pivoting data...")
    data_pivoted = data.pivot_table(
        values='resultMeanValue',
        index=['monitoringSiteIdentifier', 'phenomenonTimeReferenceYear', 'parameterWaterBodyCategory'],
        columns='observedPropertyDeterminandLabel',
        aggfunc='first'
    )
    data_pivoted = data_pivoted.reset_index()
    print(f"After pivoting: {data_pivoted.shape}")
    if 'BOD5' in data_pivoted.columns:
        data_pivoted = data_pivoted.dropna(subset=['BOD5'])
        print(f"After removing samples with missing BOD5: {data_pivoted.shape}")
    else:
        print("Warning: BOD5 column not found in the dataset!")
        potential_bod_columns = [col for col in data_pivoted.columns if 'BOD' in col]
        if potential_bod_columns:
            print(f"Potential BOD columns found: {potential_bod_columns}")
            data_pivoted.rename(columns={potential_bod_columns[0]: 'BOD5'}, inplace=True)
            data_pivoted = data_pivoted.dropna(subset=['BOD5'])
            print(f"After removing samples with missing BOD5: {data_pivoted.shape}")
        else:
            raise ValueError("BOD5 column not found and no suitable alternatives identified.")
    threshold = len(data_pivoted) * 0.1 
    data_pivoted = data_pivoted.dropna(axis=1, thresh=threshold)
    print(f"After removing features with >90% missing values: {data_pivoted.shape}")
    threshold = data_pivoted.shape[1] * 0.5  
    data_pivoted = data_pivoted.dropna(thresh=threshold)
    print(f"After removing samples with >50% missing values: {data_pivoted.shape}")
    data_pivoted = data_pivoted.fillna(data_pivoted.median())
    data_pivoted = data_pivoted[data_pivoted['parameterWaterBodyCategory'] == 'RW']
    print(f"After keeping only river water samples: {data_pivoted.shape}")
    non_numeric_cols = data_pivoted.select_dtypes(exclude=[np.number]).columns.tolist()
    cols_to_remove = [col for col in non_numeric_cols if col != 'BOD5']
    data_pivoted = data_pivoted.drop(columns=cols_to_remove)
    
    return data_pivoted

def sequential_feature_selection(X, y, estimator, n_features_to_select=None):
    n_features = X.shape[1]
    if n_features_to_select is None:
        n_features_to_select = n_features // 2 
    
    selected_features = []
    remaining_features = list(X.columns)
    best_score = -np.inf
    
    while len(selected_features) < n_features_to_select:
        best_new_score = -np.inf
        best_feature = None
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_subset = X[current_features]
            
            cv = KFold(n_splits=10, shuffle=True, random_state=42)
            scores = cross_val_score(estimator, X_subset, y, cv=cv, scoring='r2')
            mean_score = np.mean(scores)
            
            if mean_score > best_new_score:
                best_new_score = mean_score
                best_feature = feature
        
        if best_new_score > best_score:
            best_score = best_new_score
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            print(f"Added feature: {best_feature}, R² score: {best_new_score:.4f}")
        else:
            break
    
    return selected_features, best_score

def optimize_hyperparameters(model_class, param_ranges, X, y, generations=10, population_size=10):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    param_names = list(param_ranges.keys())
    
    for i, param in enumerate(param_ranges):
        param_range = param_ranges[param]
        if isinstance(param_range, tuple) and len(param_range) == 2 and all(isinstance(x, (int, float)) for x in param_range):
            if all(isinstance(x, int) for x in param_range):
                toolbox.register(f"attr_{i}", random.randint, param_range[0], param_range[1])
            else:
                toolbox.register(f"attr_{i}", random.uniform, param_range[0], param_range[1])
        elif isinstance(param_range, list):
            toolbox.register(f"attr_{i}", random.choice, param_range)
    
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                    [getattr(toolbox, f"attr_{i}") for i in range(len(param_names))], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def evaluate_params(individual):
        params = {param_names[i]: val for i, val in enumerate(individual)}
        
        if 'hidden_layer_sizes' in params and isinstance(params['hidden_layer_sizes'], (int, float)):
            hidden_sizes = tuple(int(params[f'hidden_layer_size_{i}']) for i in range(3))
            params['hidden_layer_sizes'] = hidden_sizes
            
            for i in range(3):
                if f'hidden_layer_size_{i}' in params:
                    del params[f'hidden_layer_size_{i}']
                
        model = model_class(**params)
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        return (np.mean(scores),)
    
    toolbox.register("evaluate", evaluate_params)
    toolbox.register("mate", tools.cxTwoPoint)
    
    def mutate(individual, indpb):
        for i, param in enumerate(param_names):
            param_range = param_ranges[param]
            if random.random() < indpb:
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    if all(isinstance(x, int) for x in param_range):
                        individual[i] = random.randint(param_range[0], param_range[1])
                    else:
                        individual[i] = random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    individual[i] = random.choice(param_range)
        return individual,
    
    toolbox.register("mutate", mutate, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=population_size)
    
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, 
                                      ngen=generations, stats=stats, halloffame=hof, verbose=True)
    
    best_individual = hof[0]
    best_params = {param_names[i]: val for i, val in enumerate(best_individual)}
    
    if 'hidden_layer_sizes' in best_params and isinstance(best_params['hidden_layer_sizes'], (int, float)):
        hidden_sizes = tuple(int(best_params[f'hidden_layer_size_{i}']) for i in range(3))
        best_params['hidden_layer_sizes'] = hidden_sizes
        
        for i in range(3):
            if f'hidden_layer_size_{i}' in best_params:
                del best_params[f'hidden_layer_size_{i}']
    
    return best_params, best_individual.fitness.values[0]

def calculate_feature_importance(model, X, y):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importances = result.importances_mean
    
    feature_importance = {X.columns[i]: importances[i] for i in range(len(X.columns))}
    
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_importance

def train_and_evaluate_model(X, y, model, model_name, cv=10):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    r2_scores = []
    mse_scores = []
    mae_scores = []
    relative_mse_scores = []
    relative_mae_scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        relative_mse = mse / np.var(y_test)
        relative_mae = mae / np.mean(np.abs(y_test))
        
        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)
        relative_mse_scores.append(relative_mse)
        relative_mae_scores.append(relative_mae)
    
    mean_r2 = np.mean(r2_scores)
    mean_mse = np.mean(mse_scores)
    mean_mae = np.mean(mae_scores)
    mean_relative_mse = np.mean(relative_mse_scores)
    mean_relative_mae = np.mean(relative_mae_scores)
    
    print(f"\n{model_name} Results:")
    print(f"R²: {mean_r2:.6f}")
    print(f"MSE: {mean_mse:.6f}")
    print(f"MAE: {mean_mae:.6f}")
    print(f"Relative MSE: {mean_relative_mse:.6f}")
    print(f"Relative MAE: {mean_relative_mae:.6f}")
    
    return {
        'model': model,
        'R²': mean_r2,
        'MSE': mean_mse,
        'MAE': mean_mae,
        'Relative MSE': mean_relative_mse,
        'Relative MAE': mean_relative_mae
    }

def main(file_path):
    data = preprocess_waterbase_data(file_path)
    
    X = data.drop(['BOD5', 'monitoringSiteIdentifier', 'phenomenonTimeReferenceYear', 'parameterWaterBodyCategory'], axis=1, errors='ignore')
    y = data['BOD5']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    models = {
        'RF': (
            RandomForestRegressor(random_state=42),
            {
                'n_estimators': (20, 200),
                'max_depth': (5, 50)
            }
        ),
        'SVR': (
            SVR(),
            {
                'C': (0.1, 10),
                'epsilon': (0.01, 1),
                'gamma': ['scale', 'auto']
            }
        ),
        'MLP': (
            MLPRegressor(random_state=42, max_iter=1000),
            {
                'hidden_layer_size_0': (1, 100),
                'hidden_layer_size_1': (1, 100),
                'hidden_layer_size_2': (1, 100),
                'alpha': (0.0001, 0.01),
                'learning_rate_init': (0.001, 0.1)
            }
        )
    }
    
    results = {}
    
    for model_name, (base_model, param_ranges) in models.items():
        print(f"\n{'='*50}")
        print(f"Processing {model_name} model")
        print(f"{'='*50}")
        
        print(f"\nOptimizing {model_name} hyperparameters...")
        best_params, best_score = optimize_hyperparameters(
            type(base_model), param_ranges, X_scaled, y, generations=10, population_size=10
        )
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation R² score: {best_score:.6f}")
        
        optimized_model = type(base_model)(**best_params, random_state=42 if hasattr(base_model, 'random_state') else None)
        
        print(f"\nPerforming feature selection for {model_name}...")
        selected_features, _ = sequential_feature_selection(X_scaled, y, optimized_model, n_features_to_select=10)
        print(f"Selected features: {selected_features}")
        
        X_selected = X_scaled[selected_features]
        model_results = train_and_evaluate_model(X_selected, y, optimized_model, model_name)
        
        print(f"\nCalculating feature importance for {model_name}...")
        optimized_model.fit(X_selected, y)
        feature_importance = calculate_feature_importance(optimized_model, X_selected, y)
        print("Feature importance:")
        for feature, importance in feature_importance:
            print(f"  {feature}: {importance:.6f}")
        
        model_results['Selected Features'] = selected_features
        model_results['Feature Importance'] = feature_importance
        results[model_name] = model_results
    
    best_model = max(results.keys(), key=lambda k: results[k]['R²'])
    print(f"\n\nBest model: {best_model} with R² score of {results[best_model]['R²']:.6f}")
    
    return results

if __name__ == "__main__":
    file_path = "Waterbase_v2023_1_T_WISE6_AggregatedData.csv"
    results = main(file_path)
