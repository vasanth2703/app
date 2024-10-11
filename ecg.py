import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scikit-learn.preprocessing import StandardScaler
import pandas as pd

class ECGModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ECGModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGModel(input_channels=3, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_ecg(ecg_data):
    scaler = StandardScaler()
    return scaler.fit_transform(ecg_data)

def predict_ecg(model, ecg_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    ecg_data = preprocess_ecg(ecg_data)
    ecg_tensor = torch.tensor(ecg_data, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(ecg_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    class_names = ['Normal', 'Abnormal']
    return class_names[predicted_class], probabilities[0][predicted_class].item()

def visualize_ecg(ecg_data):
    plt.figure(figsize=(12, 6))
    for i in range(ecg_data.shape[1]):
        plt.plot(ecg_data[:, i], label=f'Channel {i+1}')
    plt.title('ECG Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def generate_report(prediction, confidence, ecg_data):
    report = f"""
    ECG Analysis Report
    -------------------
    Prediction: {prediction}
    Confidence: {confidence:.2f}

    ECG Statistics:
    - Duration: {ecg_data.shape[0]} samples
    - Number of channels: {ecg_data.shape[1]}
    - Max amplitude: {np.max(ecg_data):.2f}
    - Min amplitude: {np.min(ecg_data):.2f}
    - Mean amplitude: {np.mean(ecg_data):.2f}
    - Standard deviation: {np.std(ecg_data):.2f}
    """
    return report

def main():
    model_path = r"D:\robotic project\ecg_model.pth"
    model = load_model(model_path)
    
    
    while True:
        print("\nECG Analysis Menu:")
        print("1. Load ECG data from CSV")
        print("2. Generate random ECG data (for testing)")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            file_path = input("Enter the path to your CSV file: ")
            try:
                ecg_data = pd.read_csv(file_path).values
                if ecg_data.shape[1] != 3:
                    print("Error: CSV should have exactly 3 columns for ECG channels.")
                    continue
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                continue
        elif choice == '2':
            ecg_data = np.random.randn(5000, 3)  # Generate random data for testing
        elif choice == '3':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
            continue
        
        visualize_ecg(ecg_data)
        
        prediction, confidence = predict_ecg(model, ecg_data)
        report = generate_report(prediction, confidence, ecg_data)
        
        print(report)
        
        save_choice = input("Do you want to save this report? (y/n): ")
        if save_choice.lower() == 'y':
            filename = input("Enter filename to save the report (e.g., report.txt): ")
            with open(filename, 'w') as f:
                f.write(report)
            print(f"Report saved to {filename}")

if __name__ == "__main__":
    main()
