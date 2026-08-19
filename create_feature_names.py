import joblib

feature_names = [
    "MFCC_1",
    "MFCC_2",
    "MFCC_3",
    "MFCC_4",
    "MFCC_5",
    "MFCC_6",
    "MFCC_7",
    "MFCC_8",
    "MFCC_9",
    "MFCC_10",
    "MFCC_11",
    "MFCC_12",
    "MFCC_13",
    "Zero_Crossing_Rate",
    "RMS_Energy",
    "Spectral_Centroid",
    "Spectral_Bandwidth",
    "Spectral_Rolloff",
    "Tempo",
    "Beat_Frames"
]

joblib.dump(feature_names, "models/feature_names.pkl")

print("feature_names.pkl created successfully!")
print("Number of features:", len(feature_names))