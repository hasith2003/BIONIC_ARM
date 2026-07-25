from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import get_scaled_features

def main():
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test = get_scaled_features()
    
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', random_state=42),
        "Radius Neighbors": RadiusNeighborsClassifier(radius=15.0, outlier_label=0, n_jobs=-1) 
    }
    
    results = {}
    
    print("="*60)
    print("TRAINING AND EVALUATING PAPER MODELS (Subject-Independent)")
    print("="*60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"--> {name} Test Accuracy: {acc * 100:.2f}%")

    print("\n" + "="*60)
    print("FINAL SUBJECT-INDEPENDENT COMPARISON")
    print("="*60)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<25}: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
