import os
import joblib
import numpy as np
import librosa
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def extract_mfcc(file_path, n_mfcc=13):
    """Extract MFCC features from an audio file."""
    try:
        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def load_data(real_dir, fake_dir, is_training=True):
    """Load and process audio files."""
    X = []
    y = []
    
    # Process real audio files
    print(f"Processing {'training' if is_training else 'test'} real audio files...")
    real_files = [f for f in os.listdir(real_dir) if f.endswith('.wav')]
    print(f"Found {len(real_files)} real audio files")
    for file in real_files:
        file_path = os.path.join(real_dir, file)
        print(f"Processing: {file}")
        features = extract_mfcc(file_path)
        if features is not None:
            X.append(features)
            y.append(0)  # 0 for real audio
    
    # Process fake audio files
    print(f"\nProcessing {'training' if is_training else 'test'} fake audio files...")
    fake_files = [f for f in os.listdir(fake_dir) if f.endswith('.wav')]
    print(f"Found {len(fake_files)} fake audio files")
    for file in fake_files:
        file_path = os.path.join(fake_dir, file)
        print(f"Processing: {file}")
        features = extract_mfcc(file_path)
        if features is not None:
            X.append(features)
            y.append(1)  # 1 for fake audio
    
    print(f"\nTotal {'training' if is_training else 'test'} samples: {len(X)}")
    return np.array(X), np.array(y)

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def train_model(X_train, y_train):
    """Train the SVM model."""
    print("\nTraining the model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    joblib.dump(model, 'svm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved as 'svm_model.pkl' and 'scaler.pkl'")
    
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test, dataset_name="Test"):
    """Evaluate the model on test data."""
    print(f"\nEvaluating on {dataset_name} dataset...")
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Real', 'Fake']))
    
    # Plot confusion matrix
    print(f"\nGenerating confusion matrix plot for {dataset_name} dataset...")
    plot_confusion_matrix(y_test, predictions, f'{dataset_name} Confusion Matrix')
    print(f"Confusion matrix saved as '{dataset_name.lower().replace(' ', '_')}_confusion_matrix.png'")

def main():
    # Load training data from local directories
    print("Loading training data...")
    X_train, y_train = load_data('real_audio', 'deepfake_audio', is_training=True)
    
    if len(X_train) == 0:
        print("No training data found!")
        return
    
    # Train the model
    model, scaler = train_model(X_train, y_train)
    
    # Evaluate on training data
    evaluate_model(model, scaler, X_train, y_train, "Training")
    
    # Load and evaluate on test data
    print("\nLoading test data...")
    X_test, y_test = load_data(
        'D:/Downloads/archive/for-2sec/for-2seconds/testing/real/test',
        'D:/Downloads/archive/for-2sec/for-2seconds/testing/fake/test',
        is_training=False
    )
    
    if len(X_test) == 0:
        print("No test data found!")
        return
    
    # Evaluate on test data
    evaluate_model(model, scaler, X_test, y_test, "Test")

if __name__ == "__main__":
    main() 