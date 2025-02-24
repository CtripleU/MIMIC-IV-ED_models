

def train_model(train_data, test_data, model_path):
    # Split data into features and target
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']

    # Train MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, early_stopping=True)
    mlp.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    auroc = roc_auc_score(y_test, mlp.predict_proba(X_test), multi_class="ovo")

    print(f'Accuracy: {accuracy}')
    print(f'F1-score: {f1}')
    print(f'AUROC: {auroc}')

    # Save the trained model
    with open(f'{model_path}/model.pkl', 'wb') as f:
        pickle.dump(mlp, f)

# Example usage
train_data, test_data = preprocess_data('data')
