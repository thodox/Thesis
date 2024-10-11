import numpy as np
import pandas as pd
from tqdm import tqdm 
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import regularizers
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import VotingClassifier
import datetime
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import psutil
import time
pd.set_option('display.max_columns',None)
warnings.filterwarnings('ignore')
%matplotlib inline



# XỬ LÍ DỮ LIỆU##########################################################




















    
    df_Monday= pd.read_csv("Monday-WorkingHours.pcap_ISCX.csv")
    df_Tuesday = pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv")
    df_Wednesday = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")
    df_Thursday_Morning = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    df_Thursday_Afternoon = pd.read_csv("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    df_Friday_Morning = pd.read_csv("Friday-WorkingHours-Morning.pcap_ISCX.csv")
    df_Friday_DDOS = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    df_Friday_PortScan = pd.read_csv("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")

    frames = [df_Monday, df_Tuesday, df_Wednesday,df_Thursday_Morning,df_Thursday_Afternoon,df_Friday_Morning,df_Friday_DDOS,df_Friday_PortScan]

    df = pd.concat(frames)

    """##Read data"""

    print(df.shape)

    df.drop([' Bwd PSH Flags'], axis=1, inplace=True)
    df.drop([' Bwd URG Flags'], axis=1, inplace=True)
    df.drop(['Fwd Avg Bytes/Bulk'], axis=1, inplace=True)
    df.drop([' Fwd Avg Packets/Bulk'], axis=1, inplace=True)
    df.drop([' Fwd Avg Bulk Rate'], axis=1, inplace=True)
    df.drop([' Bwd Avg Bytes/Bulk'], axis=1, inplace=True)
    df.drop([' Bwd Avg Packets/Bulk'], axis=1, inplace=True)
    df.drop(['Bwd Avg Bulk Rate'], axis=1, inplace=True)
    df.drop(['Flow Bytes/s',' Flow Packets/s'], axis=1, inplace=True)

    df.shape

    for i in df.columns:
        df = df[df[i] != "Infinity"]
        df = df[df[i] != np.nan]
        df = df[df[i] != ",,"]
    df_num = df.select_dtypes(include='number')
    df_norm = (df_num - df_num.mean()) / (df_num.max() - df_num.min())
    df[df_norm.columns] = df_norm



    df[' Label'].value_counts().plot(kind='bar')


    df[' Label'] = LabelEncoder().fit_transform(df[' Label'])

    df[' Label'].unique()


    count = np.isinf(df).values.sum()
    print(count)

    df.replace([np.inf,-np.inf], np.nan, inplace=True)

    df.dropna(axis=0, how='any', inplace=True)

    print(df.shape)

    scaler = MinMaxScaler()
    scaler_feature = scaler.fit_transform(df)
    scaler_feature.shape

    """##Split train set and test set"""

    train, test  = train_test_split(df, test_size = 0.2, random_state = 42, shuffle=True) #shuffle=False
    train[' Label'].value_counts()
    train.shape
    test.shape
    test[' Label'].value_counts()
    x_train = train.drop([' Label'], axis=1)
    y_train = train[' Label']

    x_test = test.drop([' Label'], axis=1)
    y_test = test[' Label']

    y_train[y_train != 0] = 1
    y_test[y_test != 0] = 1

#PHASE1 HUẤN LUYỆN VÀ TEST MÔ HÌNH TRÊN DỮ LIỆU GỐC #############################################################################################################################

# MÔ HÌNH LSTM 5 LAYERS###############################################################




def model_lstm(name_model, x_train, y_train):
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = f"Model: {name_model}\nTraining start time: {start_time}\n"

    # Convert DataFrame to NumPy array and reshape
    x_train_reshaped = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True, input_shape=(x_train_reshaped.shape[1], 1)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(units=256, activation='tanh', return_sequences=True),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(units=512, activation='tanh', return_sequences=True),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=False),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])

    # Logging start time
    # Training the model
    history = model.fit(x_train_reshaped, y_train, epochs=20, verbose=1)
    # Saving model
    model.save(f'{name_model}.h5')
    # Logging model summary and training history
    log_text += f"Model Summary:\n{model.summary()}\n"
    log_text += f"Training History:\n{history.history}\n"

    # Writing log to file
    with open('log.txt', 'a') as log_file:
        log_file.write(log_text)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history.history['loss'], color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(1, 21))

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(history.history['accuracy'], color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xticks(range(1, 21))

    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.title('Model LSTM Loss and Accuracy')
    plt.show()


    return model





# MÔ HÌNH DNN 6 LAYERS

from tensorflow.keras import regularizers
import datetime
def model_dnn(name_model, x_train, y_train):
    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu', input_shape=(x_train.shape[1:]),
                              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.L2(1e-4),
                              activity_regularizer=regularizers.L2(1e-5)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=128, activation='relu',
                              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.L2(1e-4),
                              activity_regularizer=regularizers.L2(1e-5)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=512, activation='relu',
                              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.L2(1e-4),
                              activity_regularizer=regularizers.L2(1e-5)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=256, activation='relu',
                              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.L2(1e-4),
                              activity_regularizer=regularizers.L2(1e-5)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=128, activation='relu',
                              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                              bias_regularizer=regularizers.L2(1e-4),
                              activity_regularizer=regularizers.L2(1e-5)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=1, activation='sigmoid'),
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    # Fit the model and record the history
    history = model.fit(x_train, y_train, epochs=5, verbose=1)
    
    # Save the model
    model.save(f'{name_model}.h5')
    
    # Log training details
    log_details = {
        'Time': str(datetime.now()),
        'Model Name': name_model,
        'Model Path': f'{name_model}.h5',
        'Training History': history.history,
    }
    
    # Write log to file
    with open('log.txt', 'a') as log_file:  # Use 'a' to append to the file
        log_file.write(str(log_details) + '\n')

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_validation_loss_mlp.png')
    plt.show()





# HUẤN LUYỆN CÁC MÔ HÌNH DT, LR, GNB VÀ GHI LOG#####################################################################




def log_training_details(message):
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    with open("log_log.txt", "a") as log_file:
        log_file.write(f"[{current_time}] {message}\n")

# Function to get current RAM usage in percentage
def get_ram_usage():
    memory_info = psutil.virtual_memory()
    return memory_info.percent

# Log initial system RAM usage
initial_ram = get_ram_usage()
log_training_details(f"Initial RAM Usage: {initial_ram}%")

# Track the time for training models
start_time = time.time()
log_training_details("Start training models...")

# Train GaussianNB model
gnb = GaussianNB()
gnb.fit(x_train, y_train)
gnb_ram = get_ram_usage()
log_training_details(f"GaussianNB RAM Usage: {gnb_ram}%")
gnb_accuracy = accuracy_score(y_test, gnb.predict(x_test))
log_training_details(f"GaussianNB Accuracy: {gnb_accuracy}")

# Train DecisionTreeClassifier with max_depth=1
dt = DecisionTreeClassifier(max_depth=1)
dt.fit(x_train, y_train)
dt_ram = get_ram_usage()
log_training_details(f"DecisionTreeClassifier(max_depth=1) RAM Usage: {dt_ram}%")
dt_accuracy = accuracy_score(y_test, dt.predict(x_test))
log_training_details(f"DecisionTreeClassifier(max_depth=1) Accuracy: {dt_accuracy}")

# Train DecisionTreeClassifier
tdt = DecisionTreeClassifier()
tdt.fit(x_train, y_train)
tdt_ram = get_ram_usage()
log_training_details(f"DecisionTreeClassifier RAM Usage: {tdt_ram}%")
tdt_accuracy = accuracy_score(y_test, tdt.predict(x_test))
log_training_details(f"DecisionTreeClassifier Accuracy: {tdt_accuracy}")

# Train LogisticRegression
lr = LogisticRegression(max_iter=200)
logic = lr.fit(x_train, y_train)
logic_ram = get_ram_usage()
log_training_details(f"LogisticRegression RAM Usage: {logic_ram}%")
logic_accuracy = accuracy_score(y_test, logic.predict(x_test))
log_training_details(f"LogisticRegression Accuracy: {logic_accuracy}")

# Save the models
joblib.dump(gnb, 'gnb.joblib')
joblib.dump(dt, 'dt.joblib')
joblib.dump(tdt, 'tdt.joblib')
joblib.dump(logic, 'logic.joblib')
log_training_details("Saved all models")

# Create and train an ensemble voting classifier
ensemble_models = [('GaussianNB', gnb), ('DecisionTreeClassifier', tdt), ('LogisticRegression', logic)]
ensemble_voting = VotingClassifier(estimators=ensemble_models, voting='hard')
ensemble_voting.fit(x_train, y_train)
ensemble_ram = get_ram_usage()
log_training_details(f"Ensemble Voting Classifier RAM Usage: {ensemble_ram}%")
ensemble_accuracy = accuracy_score(y_test, ensemble_voting.predict(x_test))
log_training_details(f"Ensemble Voting Classifier Accuracy: {ensemble_accuracy}")

# Save the ensemble model
joblib.dump(ensemble_voting, 'en_ml.joblib')
log_training_details("Saved ensemble model")

# Log total training time
end_time = time.time()
training_time = end_time - start_time
log_training_details(f"Total Training Time: {training_time} seconds")

# Log final system RAM usage
final_ram = get_ram_usage()
log_training_details(f"Final RAM Usage: {final_ram}%")



##### TÍNH TOÁN CÁC CHỈ SỐ VÀ VẼ MATRIX########################

# Function to log details
def log_details(message):
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    with open("log_log.txt", "a") as log_file:
        log_file.write(f"[{current_time}] {message}\n")

# Function to get current RAM and CPU usage
def get_system_usage():
    cpu_usage = psutil.cpu_percent(interval=None)
    memory_info = psutil.virtual_memory()
    return cpu_usage, memory_info.percent

# Function to calculate metrics and plot confusion matrix
def calculate_metrics_and_plot_cm(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    return accuracy, precision, recall, f1

# Load models and evaluate
def load_and_evaluate_models(x_test, y_test):
    models = ['gnb', 'dt', 'tdt', 'logic', 'en_ml']
    loaded_models = {}
    load_times = {}

    # Load each model
    for model_name in models:
        start_load_time = time.time()
        loaded_model = joblib.load(f'{model_name}.joblib')
        end_load_time = time.time()
        load_time = end_load_time - start_load_time
        load_times[model_name] = load_time
        loaded_models[model_name] = loaded_model
        log_details(f"Loaded {model_name} model in {load_time:.4f} seconds")

    # Evaluate each model
    for model_name, model in loaded_models.items():
        start_test_time = time.time()

        # Measure RAM and CPU usage before testing
        start_cpu, start_ram = get_system_usage()

        # Test the model
        predictions = model.predict(x_test)

        # Measure RAM and CPU usage after testing
        end_cpu, end_ram = get_system_usage()

        end_test_time = time.time()
        test_time = end_test_time - start_test_time

        # Calculate RAM and CPU usage during testing
        ram_usage = end_ram - start_ram
        cpu_usage = end_cpu - start_cpu

        # Compute metrics and plot confusion matrix
        accuracy, precision, recall, f1 = calculate_metrics_and_plot_cm(model_name, y_test, predictions)

        log_details(f"Model: {model_name}")
        log_details(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        log_details(f"Test Time: {test_time:.4f} seconds")
        log_details(f"RAM Usage for testing: {ram_usage:.2f}%, CPU Usage for testing: {cpu_usage:.2f}%")

load_and_evaluate_models(x_test, y_test)






# Function to log details
def log_details(message):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("log_log.txt", "a") as log_file:
        log_file.write(f"[{current_time}] {message}\n")

# Function to measure CPU and RAM usage
def measure_resources():
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    return cpu_usage, ram_usage

# Function to calculate metrics, plot confusion matrix, and log results
def result(model, model_1, name, x_test, y_test):
    start_time = datetime.now()
    
    # Measure initial CPU and RAM usage
    initial_cpu, initial_ram = measure_resources()

    # Perform model evaluation
    tn, fp, fn, tp = confusion_matrix(y_test, model_1.predict(x_test).round()).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    # Measure CPU and RAM usage after model evaluation
    final_cpu, final_ram = measure_resources()

    execution_time = (datetime.now() - start_time).total_seconds()

    # Log details including CPU, RAM, and execution time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_path = os.getcwd()

    log_details(f"{name}")
    log_details(f"Loss: {model[0]}")
    log_details(f"Accuracy: {accuracy}")
    log_details(f"Precision: {precision}")
    log_details(f"Recall: {recall}")
    log_details(f"F1-score: {f1}")
    log_details(f"Time: {current_time}")
    log_details(f"Path: {current_path}")
    log_details(f"CPU Usage: {initial_cpu:.2f}% -> {final_cpu:.2f}%")
    log_details(f"RAM Usage: {initial_ram:.2f}% -> {final_ram:.2f}%")
    log_details(f"Execution Time: {execution_time:.2f} seconds")

    confusion_matrix_array = [[tn, fp], [fn, tp]]
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_array, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

    return confusion_matrix_array




# HUẤN LUYỆN MÔ HÌNH DNN VÀ LSTM

model_dnn('dnn', x_train, y_train)
model_lstm('lstm', x_train, y_train)



# ĐÁNH GIÁ MÔ HÌNH DNN, LSTM VÀ ENSEMBLE DEEPLEARNING VÀ VẼ MATRIX






dnn = tf.keras.models.load_model('dnn.h5')
lstm = tf.keras.models.load_model('lstm.h5')
dnn_test = dnn.evaluate(x_test, y_test, verbose=1)
result(dnn_test, dnn , "dnn test", x_test, y_test)
lstm_test = lstm.evaluate(x_test, y_test, verbose=1)
result(lstm_test, lstm , "lstm test", x_test, y_test)



def log_details(message):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("log_log.txt", "a") as log_file:
        log_file.write(f"[{current_time}] {message}\n")

# Function to load DNN and LSTM models
def load_models():
    dnn_model = load_model('dnn.h5')
    lstm_model = load_model('lstm.h5')
    log_details("Loaded DNN and LSTM models")
    return dnn_model, lstm_model

# Function to perform sorted voting ensemble
def ensemble_voting_predict(models, x_test):
    dnn_model, lstm_model = models
    dnn_predictions = dnn_model.predict(x_test)
    lstm_predictions = lstm_model.predict(x_test)
    
    # Perform sorted voting
    final_predictions = (dnn_predictions + lstm_predictions) / 2  # Simple average for binary classification
    
    return final_predictions

# Function to calculate metrics and plot confusion matrix
def calculate_metrics_and_plot_cm(y_true, y_pred, name):
    # Tính các chỉ số đánh giá
    accuracy = accuracy_score(y_true, y_pred.round())
    precision = precision_score(y_true, y_pred.round())
    recall = recall_score(y_true, y_pred.round())
    f1 = f1_score(y_true, y_pred.round())

    # Tính ma trận nhầm lẫn và chỉ giữ lại giá trị fn và tp
    cm = confusion_matrix(y_true, y_pred.round())
    fn, tp = cm[1, 0], cm[1, 1]
    cm_modified = np.array([[0, 0], [fn, tp]])

    # Vẽ ma trận nhầm lẫn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_modified, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title('Confusion Matrix for Ensemble Deep Learning Model')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    # Lưu ma trận nhầm lẫn vào tập tin hình ảnh
    current_path = os.getcwd()
    filename = os.path.join(current_path, f'confusion_matrix_{name}.png')
    plt.savefig(filename)
    plt.close()

    return accuracy, precision, recall, f1

# Function to log results
def log_results(model_name, y_true, y_pred):
    accuracy, precision, recall, f1 = calculate_metrics_and_plot_cm(y_true, y_pred)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_path = os.getcwd()

    log_details(f"Ensemble Model: {model_name}")
    log_details(f"Accuracy: {accuracy:.4f}")
    log_details(f"Precision: {precision:.4f}")
    log_details(f"Recall: {recall:.4f}")
    log_details(f"F1-score: {f1:.4f}")
    log_details(f"Time: {current_time}")
    log_details(f"Path: {current_path}")


#ĐÁNH GIÁ ENSEMBLE DEEPLEARNING
dnn_model, lstm_model = load_models()
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), x_test)
log_results("Sorted Voting Ensemble Deep Learning", y_test, ensemble_dl)




#PHASE2 TẤN CÔNG ĐỐI KHÁNG#########################################################################################################
# TẤN CÔNG ĐỐI KHÁNG







with open('log_log.txt', 'a') as f:
    f.write("\nFGSM ATTACK:\n")


# LẤY CÁC MẪU ĐỘC HẠI, LOẠI BỎ CÁC MẪU BÌNH THƯỜNG####################################

mask_nonadv = y_test == 0

# Tạo mặt nạ để chọn các phần tử có nhãn là 1 (adversarial)
mask_adv = y_test == 1

# Tách dữ liệu và nhãn theo mặt nạ
x_noadv = x_test[mask_nonadv]
y_noadv = y_test[mask_nonadv]

x_adv = x_test[mask_adv]
y_adv = y_test[mask_adv]



#FGSM ATTACK----------------------------------------------------------------------------



def fgsm_attack(model, x_test, y_test, epsilon):
    # Bắt đầu đo thời gian tấn công
    start_time = datetime.now()
    
    # Tính phần trăm RAM sử dụng ban đầu
    initial_ram_percent = psutil.virtual_memory().percent
    
    # Chuyển đổi dữ liệu x_test thành tensor TensorFlow
    x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
    
    # Sử dụng GradientTape để tính gradient của loss theo dữ liệu đầu vào
    with tf.GradientTape() as tape:
        tape.watch(x_test_tensor)
        predictions = model(x_test_tensor)
        predictions = tf.reshape(predictions, [-1])  
        loss = tf.keras.losses.binary_crossentropy(y_test, predictions, from_logits=True)
    
    # Tính gradient của loss theo dữ liệu đầu vào
    gradient = tape.gradient(loss, x_test_tensor)
    
    # Tính trung bình và độ lệch chuẩn của gradient
    gradient_mean = tf.reduce_mean(gradient)
    gradient_std = tf.math.reduce_std(gradient)
    
    # Tính signed gradient (gradient với dấu)
    signed_grad = tf.sign(gradient)
    
    # Tạo dữ liệu tấn công bằng cách dịch chuyển x_test theo hướng signed_grad với độ lớn epsilon
    x_test_adv = x_test_tensor + epsilon * signed_grad
    
    # Giới hạn giá trị của dữ liệu tấn công trong khoảng từ 0 đến 1 để đảm bảo là ảnh đầu ra hợp lệ
    x_test_adv = tf.clip_by_value(x_test_adv, 0, 1)
    
    # Chuyển đổi dữ liệu tấn công từ tensor TensorFlow sang numpy array
    x_test_adv_numpy = x_test_adv.numpy()
    
    # Chuyển đổi dữ liệu y_test từ tensor TensorFlow sang numpy array nếu y_test là một tensor
    y_test_numpy = y_test.numpy() if isinstance(y_test, tf.Tensor) else y_test
    
    # Mở rộng y_test thành mảng 2D để concatenate với dữ liệu tấn công
    y_test_numpy = np.expand_dims(y_test_numpy, axis=1)
    
    # Ghép dữ liệu tấn công (x_test_adv) với nhãn (y_test_numpy) để tạo thành một mảng tổng hợp adversarial_data
    adversarial_data = np.concatenate((x_test_adv_numpy, y_test_numpy), axis=1)
    
    # Tạo DataFrame từ adversarial_data với các tên cột là feature_{i} và label
    df = pd.DataFrame(adversarial_data, columns=[f'feature_{i}' for i in range(x_test_adv_numpy.shape[1])] + ['label'])
    
    # Lưu vào CSV với tên chứa giá trị epsilon
    file_name = f'fgsm_epsilon_{epsilon}.csv'
    df.to_csv(file_name, index=False)
    
    # Tính toán thời gian kết thúc tấn công và các chỉ số RAM, CPU
    end_time = datetime.now()
    final_ram_percent = psutil.virtual_memory().percent
    cpu_usage_percent = psutil.cpu_percent()
    
    # Ghi log vào file log_log.txt
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"Attack started at {start_time}, RAM usage: {initial_ram_percent}%, Data load: {final_ram_percent}%, CPU usage: {cpu_usage_percent}%, Epsilon: {epsilon}, Gradient Mean: {gradient_mean.numpy()}, Gradient Std: {gradient_std.numpy()}"
    
    with open("log_log.txt", "a") as log_file:
        log_file.write(f"[{current_time}] {log_message}\n")
    
    # Trả về dữ liệu tấn công đã được tạo
    return x_test_adv


epsilon = [0.01,0.02,0.05,0.1,0.2]

x_fgsm = fgsm_attack(dnn, x_adv, y_adv, epsilon)


#ZOO ATTACK------------------------------------------------------------------------------------




with open('log_log.txt', 'a') as f:
    f.write("\nZOO ATTACK:\n")




def zooattack(model, x_test, y_test, max_iter, epsilon=0.01, num_samples=50, sigma=1e-3):
    def zoo_loss(probas, y_true):
        return tf.reduce_sum(tf.square(probas - y_true))
    
    def estimate_gradient(model, x, y_true, num_samples, sigma):
        gradients = np.zeros_like(x)
        for i in range(num_samples):
            noise = np.random.normal(size=x.shape)
            noise = noise / np.linalg.norm(noise)
            
            # Đánh giá hàm mất mát (loss) khi thêm nhiễu
            probas_plus = model(x + sigma * noise, training=False)
            probas_minus = model(x - sigma * noise, training=False)
            
            # Tính toán loss với và không có nhiễu
            loss_plus = zoo_loss(probas_plus, y_true)
            loss_minus = zoo_loss(probas_minus, y_true)
            
            # Ước tính gradient bằng finite differences
            gradients += (loss_plus - loss_minus) / (2 * sigma) * noise
        
        gradients /= num_samples
        return gradients

    # Chuyển đổi x_test và y_test thành tensors nếu chưa phải
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_true = tf.one_hot(y_test, depth=np.unique(y_test).shape[0])
    
    adversarial_examples = np.zeros_like(x_test)

    # Thiết lập ghi log
    log_filename = 'log_log.txt'
    with open(log_filename, 'a') as log_file:
        log_file.write(f'ZOO ATTACK - {datetime.now()}\n')  # Ghi lại thời gian bắt đầu tấn công ZOO
        log_file.write(f'Max iterations: {max_iter}, Epsilon: {epsilon}, Num samples: {num_samples}, Sigma: {sigma}\n')  # Ghi lại các tham số của tấn công ZOO

    # Giám sát sử dụng RAM và CPU ban đầu
    initial_ram_percent = psutil.virtual_memory().percent
    initial_cpu_percent = psutil.cpu_percent()
    
    # Hiển thị thanh tiến trình
    progress_bar = tqdm(total=max_iter, desc='ZOO Attack: ', unit='iteration', dynamic_ncols=True)
    for i in range(max_iter):
        gradients = estimate_gradient(model, x_test, y_true, num_samples, sigma)
        perturbation = np.sign(gradients)
        x_test = np.clip(x_test + epsilon * perturbation, 0, 1)
        
        probas = model(tf.convert_to_tensor(x_test, dtype=tf.float32), training=False)
        loss = zoo_loss(probas, y_true)
        
        # Ghi log chi tiết từng lần lặp
        with open(log_filename, 'a') as log_file:
            log_file.write(f'Iteration {i}, Loss: {loss.numpy()}\n')

        # Cập nhật thanh tiến trình
        progress_bar.set_postfix(Iteration=i, Loss=loss.numpy())
        progress_bar.update(1)

    progress_bar.close()

    # Lưu các ví dụ adversarial và nhãn tương ứng vào file CSV
    csv_filename = f"zoo_{max_iter}.csv"
    labels = np.argmax(y_true, axis=1)
    data = np.column_stack((x_test.reshape(x_test.shape[0], -1), labels))
    pd.DataFrame(data).to_csv(csv_filename, header=False, index=False)

    # Giám sát sử dụng RAM và CPU cuối cùng
    final_ram_percent = psutil.virtual_memory().percent
    final_cpu_percent = psutil.cpu_percent()
      
    # Ghi log RAM, CPU và thời gian tấn công vào file
    with open(log_filename, 'a') as log_file:
        log_file.write(f"Initial RAM Usage: {initial_ram_percent}%, Initial CPU Usage: {initial_cpu_percent}%\n")
        log_file.write(f"Final RAM Usage: {final_ram_percent}%, Final CPU Usage: {final_cpu_percent}%\n")
    
    return x_test



max_iter = [10,50,100,200]
x_zoo = zooattack(dnn, x_adv, y_adv, max_iter=10)








#DEEPFOOL ATTACK------------------------------------------------------------------------------------



def deepfool_attack(model, x_test, y_test, max_iter, epsilon=0.01):
    # Định nghĩa hàm loss sử dụng trong DeepFool Attack
    def deepfool_loss(probas, y_true):
        return tf.reduce_sum(tf.square(probas - y_true))

    # Khởi tạo biến x_adv là một biến TensorFlow có giá trị ban đầu là x_test
    x_adv = tf.Variable(x_test, dtype=tf.float32)
    # One-hot encode nhãn y_test
    y_true = tf.one_hot(y_test, depth=np.unique(y_test).shape[0])

    # Đặt tên file log và ghi thông tin chi tiết về cuộc tấn công
    log_filename = 'log_log.txt'
    log_data = {
        'timestamp': str(datetime.now()),
        'attack': 'DeepFool Attack',
        'epsilon': epsilon
    }
    
    with open(log_filename, 'a') as log_file:
        log_file.write(f"Timestamp: {log_data['timestamp']}\n")
        log_file.write(f"Attack: {log_data['attack']}\n")
        log_file.write(f"Epsilon: {log_data['epsilon']}\n\n")

    # Khởi tạo thanh tiến trình để theo dõi tiến độ của cuộc tấn công
    progress_bar = tqdm(total=max_iter)
    for i in range(max_iter):
        # Tính toán gradient của loss theo x_adv
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            probas = model(x_adv, training=False)
            loss = deepfool_loss(probas, y_true)

        gradients = tape.gradient(loss, x_adv)
        # Tính toán độ lớn của perturbation và điều chỉnh x_adv
        perturbation = tf.sign(gradients)
        x_adv = tf.clip_by_value(x_adv + epsilon * perturbation, 0, 1)

        # Cập nhật tiến độ trên thanh tiến trình sau mỗi 10 iterations
        if i % 10 == 0:
            progress_bar.set_description(f"Iteration {i}, Loss: {loss.numpy()}")
            progress_bar.update(10)

        # Ghi log chi tiết cho mỗi iteration
        log_data_iter = {
            'iteration': i,
            'loss': loss.numpy()
        }

        with open(log_filename, 'a') as log_file:
            log_file.write(f"Iteration {log_data_iter['iteration']}, Loss: {log_data_iter['loss']}\n")

        # Tính toán và ghi log sử dụng CPU và RAM
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent

        with open(log_filename, 'a') as log_file:
            log_file.write(f"Iteration {log_data_iter['iteration']}, CPU Usage: {cpu_usage}%\n")
            log_file.write(f"Iteration {log_data_iter['iteration']}, RAM Usage: {ram_usage}%\n")

    progress_bar.close()

    # Lưu các ví dụ phản đối và nhãn tương ứng vào file CSV
    csv_filename = f"deepfool_{max_iter}.csv"
    adversarial_examples = x_adv.numpy()
    labels = tf.argmax(y_true, axis=1).numpy()
    data = np.column_stack((adversarial_examples.reshape(adversarial_examples.shape[0], -1), labels))
    pd.DataFrame(data).to_csv(csv_filename, header=False, index=False)

    return adversarial_examples




max_iter= [10,50,100,200]
x_df= deepfool_attack(dnn, x_adv, y_adv, max_iter=10) 














#CW ATTACK ------------------------------------------------------------------------------------------------------------------------
def carlini_wagner(model, x_test, y_test, max_iter, learning_rate=0.01, confidence=1):
    def cw_loss(logits, labels, confidence):
        correct_logit = tf.reduce_sum(labels * logits, axis=1)
        wrong_logit = tf.reduce_max((1 - labels) * logits - labels * 1e4, axis=1)
        return tf.maximum(correct_logit - wrong_logit + confidence, 0)
    
    def perturb_input(x, perturbation, clip_min, clip_max):
        return tf.clip_by_value(x + perturbation, clip_min, clip_max)

    # Ensure x_test and y_test are tensors
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_true = tf.one_hot(y_test, depth=np.unique(y_test).shape[0])

    # Initialize the perturbation variable
    perturbation = tf.Variable(tf.zeros_like(x_test), dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate)

    # Create a progress bar
    progress_bar = tqdm(total=max_iter, desc='Carlini-Wagner Attack', unit='iteration', dynamic_ncols=True)

    # Logging details
    log_filename = 'log_log.txt'  # Changed log file name to log_log.txt
    log_data = {
        'timestamp': str(datetime.now()),
        'attack': 'Carlini-Wagner Attack',
        'max_iterations': max_iter,
        'learning_rate': learning_rate,
        'confidence': confidence
    }

    with open(log_filename, 'a') as log_file:
        log_file.write(str(log_data) + '\n')

    for i in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(perturbation)
            x_perturbed = perturb_input(x_test, perturbation, 0, 1)
            logits = model(x_perturbed, training=False)
            loss = tf.reduce_sum(cw_loss(logits, y_true, confidence))

        gradients = tape.gradient(loss, perturbation)
        optimizer.apply_gradients([(gradients, perturbation)])

        progress_bar.set_postfix(Iteration=i, Loss=loss.numpy())
        progress_bar.update(1)

        # Log loss for each iteration
        log_data_iter = {
            'iteration': i,
            'loss': loss.numpy()
        }

        # Calculate CPU and RAM usage percentages
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent

        log_data_iter['cpu_usage'] = cpu_usage
        log_data_iter['ram_usage'] = ram_usage

        with open(log_filename, 'a') as log_file:
            log_file.write(str(log_data_iter) + '\n')

    progress_bar.close()

    # Obtain the final adversarial examples
    adversarial_examples = perturb_input(x_test, perturbation, 0, 1).numpy()

    # Save adversarial examples and their labels to a CSV file
    csv_filename = f"cw_{max_iter}.csv"
    labels = np.argmax(y_true, axis=1)
    data = np.column_stack((adversarial_examples.reshape(adversarial_examples.shape[0], -1), labels))
    pd.DataFrame(data).to_csv(csv_filename, header=False, index=False)

    return adversarial_examples






max_iter = [10,50,100,200]
x_cw= carlini_wagner(dnn, x_adv, y_adv, max_iter=10)  

#-------------------------------------- PHASE 4: THU NGHIEM MAU DOI KHANG --------------------------------------
#FGSM
X_fgsm = np.concatenate((x_fgsm, x_noadv),  axis=0)
Y_fgsm = np.concatenate((y_adv, y_noadv),  axis=0)
detectionrate_ml(gnb, "GaussianNB testing FGSM attack", X_fgsm, Y_fgsm)
detectionrate_ml(tdt, "DT testing FGSM Attack", X_fgsm, Y_fgsm)
detectionrate_ml(logic, "LogisticRegression testing FGSM attack", X_fgsm, Y_fgsm)
detectionrate_ml(ensemble_voting, "Ensemble ML testing FGSM attack", X_fgsm, Y_fgsm)
detectionrate_dl(dnn_test, dnn , "DNN testing FGSM attack", X_fgsm, Y_fgsm)
detectionrate_dl(lstm_test, lstm , "LSTM testing FGSM attack", X_fgsm, Y_fgsm)
dnn_model, lstm_model = load_models()
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_fgsm)
log_results("Ensemble Deep Learning  testing FGSM attack ", Y_fgsm, ensemble_dl)


#DF-----------------
x_df = np.concatenate((x_df, x_noadv), axis=0)
y_df = np.concatenate((y_adv, y_noadv), axis=0)

detectionrate_ml(gnb, "GaussianNB testing DF", x_df, y_df)
detectionrate_ml(tdt, "DT testing DF", x_df, y_df)
detectionrate_ml(logic, "LogisticRegression testing DF", x_df, y_df)
detectionrate_ml(ensemble_voting, "Ensemble ML testing DF", x_df, y_df)
detectionrate_dl(dnn_test, dnn, "DNN testing DF", x_df, y_df)
detectionrate_dl(lstm_test, lstm, "LSTM testing DF", x_df, y_df)
dnn_model, lstm_model = load_models()
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), x_df)
log_results("Ensemble Deep Learning testing DF", y_df, ensemble_dl)

#CW--------------------------------


X_cw = np.concatenate((x_cw, x_noadv), axis=0)
Y_cw = np.concatenate((y_adv, y_noadv), axis=0)

detectionrate_ml(gnb, "GaussianNB testing CW", X_cw, Y_cw)
detectionrate_ml(tdt, "DT testing CW", X_cw, Y_cw)
detectionrate_ml(logic, "LogisticRegression testing CW", X_cw, Y_cw)
detectionrate_ml(ensemble_voting, "Ensemble ML testing CW", X_cw, Y_cw)
detectionrate_dl(dnn_test, dnn, "DNN testing CW", X_cw, Y_cw)
detectionrate_dl(lstm_test, lstm, "LSTM testing CW", X_cw, Y_cw)

dnn_model, lstm_model = load_models()
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_cw)
log_results("Ensemble Deep Learning testing CW", Y_cw, ensemble_dl)




X_zoo = np.concatenate((x_zoo, x_noadv), axis=0)
Y_zoo = np.concatenate((y_adv, y_noadv), axis=0)

detectionrate_ml(gnb, "GaussianNB testing ZOO", X_zoo, Y_zoo)
detectionrate_ml(tdt, "DT testing ZOO", X_zoo, Y_zoo)
detectionrate_ml(logic, "LogisticRegression testing ZOO", X_zoo, Y_zoo)
detectionrate_ml(ensemble_voting, "Ensemble ML testing ZOO", X_zoo, Y_zoo)
detectionrate_dl(dnn_test, dnn, "DNN testing ZOO", X_zoo, Y_zoo)
detectionrate_dl(lstm_test, lstm, "LSTM testing ZOO", X_zoo, Y_zoo)

dnn_model, lstm_model = load_models()
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_zoo)
log_results("Ensemble Deep Learning testing ZOO", Y_zoo, ensemble_dl)






























#Ơ---------------------------------------- PHASE 5: HUAN LUYEN PHONG THU DOI KHANG --------------------------------------------------------





def detectionrate_ml(model, name, X_test, y_test):
    actual = y_test
    predicted = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    tn, fp, fn, tp = confusion_matrix.ravel()

    recall = tp / (tp + fn)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_path = os.getcwd()
    # Plot confusion matrix
    confusion_matrix_array = [[0, 0], [fn, tp]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {name}')
    plt.savefig(f'confusion_matrix_{name}.png')  # Save confusion matrix plot
    plt.show()

    # Calculate and log CPU and RAM usage
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent

    with open('log_log.txt', 'a') as log_file:
        log_file.write(f'{name}\n')
        log_file.write(f'DETECTION RATES: {recall}\n')
        log_file.write(f'Time: {current_time}\n')
        log_file.write(f'CPU Usage: {cpu_usage}%\n')
        log_file.write(f'RAM Usage: {ram_usage}%\n\n')

    return confusion_matrix






#----------------------- tranning --------------------------------------------------------------



def train(x_train, y_train, name):
    def log_training_details(message):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        with open("log_log.txt", "a") as log_file:
            log_file.write(f"[{current_time}] {message}\n")

    # Function to get current RAM usage in percentage
    def get_ram_usage():
        memory_info = psutil.virtual_memory()
        return memory_info.percent

    # Log initial system RAM usage
    initial_ram = get_ram_usage()
    log_training_details(f"Initial RAM Usage: {initial_ram}%")

    # Track the time for training models
    start_time = time.time()
    log_training_details("Start training models...")

    # Train GaussianNB model
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    gnb_ram = get_ram_usage()
    log_training_details(f"GaussianNB RAM Usage: {gnb_ram}%")
    gnb_accuracy = accuracy_score(y_test, gnb.predict(x_test))
    log_training_details(f"GaussianNB Accuracy: {gnb_accuracy}")

    # Train DecisionTreeClassifier with max_depth=1
    dt = DecisionTreeClassifier(max_depth=1)
    dt.fit(x_train, y_train)
    dt_ram = get_ram_usage()
    log_training_details(f"DecisionTreeClassifier(max_depth=1) RAM Usage: {dt_ram}%")
    dt_accuracy = accuracy_score(y_test, dt.predict(x_test))
    log_training_details(f"DecisionTreeClassifier(max_depth=1) Accuracy: {dt_accuracy}")

    # Train DecisionTreeClassifier
    tdt = DecisionTreeClassifier()
    tdt.fit(x_train, y_train)
    tdt_ram = get_ram_usage()
    log_training_details(f"DecisionTreeClassifier RAM Usage: {tdt_ram}%")
    tdt_accuracy = accuracy_score(y_test, tdt.predict(x_test))
    log_training_details(f"DecisionTreeClassifier Accuracy: {tdt_accuracy}")

    # Train LogisticRegression
    lr = LogisticRegression(max_iter=200)
    logic = lr.fit(x_train, y_train)
    logic_ram = get_ram_usage()
    log_training_details(f"LogisticRegression RAM Usage: {logic_ram}%")
    logic_accuracy = accuracy_score(y_test, logic.predict(x_test))
    log_training_details(f"LogisticRegression Accuracy: {logic_accuracy}")

    # Save the models
    joblib.dump(gnb, 'gnb_fgsm.joblib')
    joblib.dump(dt, 'dt_fgsm.joblib')
    joblib.dump(tdt, 'tdt_fgsm.joblib')
    joblib.dump(logic, 'logic_fgsm.joblib')
    log_training_details("Saved all models")

    # Create and train an ensemble voting classifier
    ensemble_models = [('GaussianNB', gnb), ('DecisionTreeClassifier', tdt), ('LogisticRegression', logic)]
    ensemble_voting = VotingClassifier(estimators=ensemble_models, voting='hard')
    ensemble_voting.fit(x_train, y_train)
    ensemble_ram = get_ram_usage()
    log_training_details(f"Ensemble Voting Classifier RAM Usage: {ensemble_ram}%")
    ensemble_accuracy = accuracy_score(y_test, ensemble_voting.predict(x_test))
    log_training_details(f"Ensemble Voting Classifier Accuracy: {ensemble_accuracy}")

    # Save the ensemble model
    joblib.dump(ensemble_voting, 'en_ml_fgsm.joblib')
    log_training_details("Saved ensemble model")

    # Log total training time
    end_time = time.time()
    training_time = end_time - start_time
    log_training_details(f"Total Training Time: {training_time} seconds")

    # Log final system RAM usage
    final_ram = get_ram_usage()
    log_training_details(f"Final RAM Usage: {final_ram}%")  






#--------------------- chia lai du lieu---------------------------
train, test  = train_test_split(df, test_size = 0.2, random_state = 42, shuffle=True) #shuffle=False

train[' Label'].value_counts()

train.shape

test.shape

test[' Label'].value_counts()

x_train = train.drop([' Label'], axis=1)
y_train = train[' Label']

x_test = test.drop([' Label'], axis=1)
y_test = test[' Label']

y_train[y_train != 0] = 1
y_test[y_test != 0] = 1



#------------------- THEM DU LIEU DOI KHANG VAO DU LIEU GOC DE TRAIN------------------------
x_train_fgsm = np.concatenate((x_fgsm, x_train), axis=0)
y_train_fgsm = np.concatenate((y_adv, y_train), axis=0)
x_train_df = np.concatenate((x_df, x_train), axis=0)
y_train_df = np.concatenate((y_adv, y_train), axis=0)
x_train_cw = np.concatenate((x_cw, x_train), axis=0)
y_train_cw = np.concatenate((y_adv, y_train), axis=0)
x_train_zoo = np.concatenate((x_zoo, x_train), axis=0)
y_train_zoo = np.concatenate((y_adv, y_train), axis=0)
















lr_fgsm = LogisticRegression(max_iter=200)
logic_fgsm = lr_fgsm.fit(x_train_fgsm, y_train_fgsm)

lr_df = LogisticRegression(max_iter=200)
logic_df = lr_df.fit(x_train_df, y_train_df)

lr_cw = LogisticRegression(max_iter=200)
logic_cw = lr_cw.fit(x_train_cw, y_train_cw)

lr_zoo = LogisticRegression(max_iter=200)
logic_zoo = lr_zoo.fit(x_train_zoo, y_train_zoo)

# Save models using joblib
joblib.dump(logic_fgsm, 'logic_fgsm.joblib')
joblib.dump(logic_df, 'logic_df.joblib')
joblib.dump(logic_cw, 'logic_cw.joblib')
joblib.dump(logic_zoo, 'logic_zoo.joblib')






def a(gnb, tdt, logic, x_train, y_train, name):
    ensemble_models = [('GaussianNB', gnb), ('DecisionTreeClassifier', tdt), ('LogisticRegression', logic)]
    ensemble_voting = VotingClassifier(estimators=ensemble_models, voting='hard')
    ensemble_voting.fit(x_train, y_train)

    # Save the ensemble model
    joblib.dump(ensemble_voting, f'{name}.joblib')





#----------------------- LOAD CAC MO HINH DA TRAIN ----------------------------------------





def log_load_model(name, model_path):
    start_time = datetime.now()
    model = joblib.load(model_path) if model_path.endswith('.joblib') else tf.keras.models.load_model(model_path)
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_path = os.getcwd()

    with open('log_log.txt', 'a') as log_file:
        log_file.write(f"{name}:\n")
        log_file.write(f"Time: {current_time}\n")
        log_file.write(f"Path: {current_path}\n")
        log_file.write(f"Execution Time: {execution_time} seconds\n")

    return model
gnb_fgsm = log_load_model('gnb_fgsm', 'gnb_fgsm.joblib')
tdt_fgsm = log_load_model('tdt_fgsm', 'tdt_fgsm.joblib')
logic_fgsm = log_load_model('logic_fgsm', 'logic_fgsm.joblib')
dnn_fgsm = log_load_model('dnn_fgsm', 'dnn_fgsm.h5')
lstm_fgsm = log_load_model('lstm_fgsm', 'lstm_fgsm.h5')
en_ml_fgsm = log_load_model('en_ml_fgsm', 'en_ml_fgsm.joblib')

# Example usage for cw models
gnb_cw = log_load_model('gnb_cw', 'gnb_cw.joblib')
tdt_cw = log_load_model('tdt_cw', 'tdt_cw.joblib')
logic_cw = log_load_model('logic_cw', 'logic_cw.joblib')
dnn_cw = log_load_model('dnn_cw', 'dnn_cw.h5')
lstm_cw = log_load_model('lstm_cw', 'lstm_cw.h5')
en_ml_cw = log_load_model('en_ml_cw', 'en_ml_cw.joblib')

# Example usage for zoo models
gnb_zoo = log_load_model('gnb_zoo', 'gnb_zoo.joblib')
tdt_zoo = log_load_model('tdt_zoo', 'tdt_zoo.joblib')
logic_zoo = log_load_model('logic_zoo', 'logic_zoo.joblib')
dnn_zoo = log_load_model('dnn_zoo', 'dnn_zoo.h5')
lstm_zoo = log_load_model('lstm_zoo', 'lstm_zoo.h5')
en_ml_zoo = log_load_model('en_ml_zoo', 'en_ml_zoo.joblib')

# Example usage for df models
gnb_df = log_load_model('gnb_df', 'gnb_df.joblib')
tdt_df = log_load_model('tdt_df', 'tdt_df.joblib')
logic_df = log_load_model('logic_df', 'logic_df.joblib')
dnn_df = log_load_model('dnn_df', 'dnn_df.h5')
lstm_df = log_load_model('lstm_df', 'lstm_df.h5')
en_ml_df = log_load_model('en_ml_df', 'en_ml_df.joblib')







#------------------------------------ 




mask_nonadv = y_test == 0

# Tạo mặt nạ để chọn các phần tử có nhãn là 1 (adversarial)
mask_adv = y_test == 1

# Tách dữ liệu và nhãn theo mặt nạ
x_noadv = x_test[mask_nonadv]
y_noadv = y_test[mask_nonadv]

x_adv = x_test[mask_adv]
y_adv = y_test[mask_adv]




# Function to load DNN and LSTM models
def load_models(name1,name2):
    dnn_model = load_model(name1)
    lstm_model = load_model(name2)
    log_details("Loaded DNN and LSTM models")
    return dnn_model, lstm_model





#------------------------ KIEM TRA DOI KHANG VA TEST TINH NHAN BIẾT KHI CHƯA BIẾT TRƯỚC CÁC MẪU TẤN CÔNG KHÁC --------------------------------------

#TAO DU LIEU DOI KHANG MOI



x_cw= carlini_wagner(dnn, x_adv, y_adv, max_iter=10)  
x_df= deepfool_attack(dnn, x_adv, y_adv, max_iter=10) 
x_fgsm = fgsm_attack(dnn, x_adv, y_adv, epsilon=0.01)
x_zoo = zooattack(dnn, x_adv, y_adv, max_iter=10)






















#---- ZOO ATTACK 





X_zoo = np.concatenate((x_zoo, x_noadv), axis=0)
Y_zoo = np.concatenate((y_adv, y_noadv), axis=0)

detectionrate_ml(gnb_zoo, "GaussianNB defense zoo testing Zoo attack", X_zoo, Y_zoo)
detectionrate_ml(tdt_zoo, "DT testing defense zoo Zoo Attack", X_zoo, Y_zoo)
detectionrate_ml(logic_zoo, "LogisticRegression defense zoo testing Zoo attack", X_zoo, Y_zoo)
detectionrate_ml(en_ml_zoo, "Ensemble ML defense zoo testing Zoo attack", X_zoo, Y_zoo)
detectionrate_dl(dnn_test, dnn_zoo , "DNN defense zoo testing Zoo attack", X_zoo, Y_zoo)
detectionrate_dl(lstm_test, lstm_zoo , "LSTM defense zoo testing Zoo attack", X_zoo, Y_zoo)
dnn_model, lstm_model = load_models('dnn_zoo.h5','lstm_zoo.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_zoo)
log_results("Ensemble Deep Learning testing Zoo attack ", Y_zoo, ensemble_dl)


detectionrate_ml(gnb_zoo, "GaussianNB defense zoo testing before FGSM attack", X_fgsm, Y_fgsm)
detectionrate_ml(tdt_zoo, "DT testing defense zoo testing before FGSM Attack", X_fgsm, Y_fgsm)
detectionrate_ml(logic_zoo, "LogisticRegression defense zoo testing before FGSM attack", X_fgsm, Y_fgsm)
detectionrate_ml(en_ml_zoo, "Ensemble ML defense zoo testing before FGSM attack", X_fgsm, Y_fgsm)
detectionrate_dl(dnn_test, dnn_zoo , "DNN defense zoo testing before FGSM attack",X_fgsm, Y_fgsm)
detectionrate_dl(lstm_test, lstm_zoo , "LSTM defense zoo testing before FGSM attack", X_fgsm, Y_fgsm)
dnn_model, lstm_model = load_models('dnn_zoo.h5','lstm_zoo.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_fgsm)
log_results("Ensemble Deep Learning testing Zoo before FGSM attack ", Y_fgsm, ensemble_dl)



detectionrate_ml(gnb_zoo, "GaussianNB defense zoo testing before Deepfool attack", X_df, Y_df)
detectionrate_ml(tdt_zoo, "DT testing defense zoo testing before Deepfool  Attack", X_df, Y_df)
detectionrate_ml(logic_zoo, "LogisticRegression defense zoo testing before Deepfool attack", X_df, Y_df)
detectionrate_ml(en_ml_zoo, "Ensemble ML defense zoo testing before Deepfool attack", X_df, Y_df)
detectionrate_dl(dnn_test, dnn_zoo , "DNN defense zoo testing before Deepfool attack",X_df, Y_df)
detectionrate_dl(lstm_test, lstm_zoo , "LSTM defense zoo testing before Deepfool attack", X_df, Y_df)
dnn_model, lstm_model = load_models('dnn_zoo.h5','lstm_zoo.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_df)
log_results("Ensemble Deep Learning defense zoo testing Deepfool attack ", Y_df, ensemble_dl)



detectionrate_ml(gnb_zoo, "GaussianNB defense zoo testing before CW attack", X_cw, Y_cw)
detectionrate_ml(tdt_zoo, "DT testing defense testing before CW  Attack",X_cw, Y_cw)
detectionrate_ml(logic_zoo, "LogisticRegression defense zoo testing before CW attack", X_cw, Y_cw)
detectionrate_ml(en_ml_zoo, "Ensemble ML defense zoo testing before CW attack", X_cw, Y_cw)
detectionrate_dl(dnn_test, dnn_zoo , "DNN defense zoo testing before CW attack",X_cw, Y_cw)
detectionrate_dl(lstm_test, lstm_zoo , "LSTM defense zoo testing before CW attack",X_cw, Y_cw)
dnn_model, lstm_model = load_models('dnn_zoo.h5','lstm_zoo.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_cw)
log_results("Ensemble Deep Learning defense zoo testing CW attack ", Y_cw, ensemble_dl)




#----------------------- FGSM ATTACK------------------------




detectionrate_ml(gnb_fgsm, "GaussianNB defense FGSM testing FGSM attack", X_fgsm, Y_fgsm)
detectionrate_ml(tdt_fgsm, "DT testing defense FGSM FGSM Attack", X_fgsm, Y_fgsm)
detectionrate_ml(logic_fgsm, "LogisticRegression defense FGSM testing FGSM attack", X_fgsm, Y_fgsm)
detectionrate_ml(en_ml_fgsm, "Ensemble ML defense FGSM testing FGSM attack", X_fgsm, Y_fgsm)
detectionrate_dl(dnn_test, dnn_fgsm , "DNN defense FGSM testing FGSM attack", X_fgsm, Y_fgsm)
detectionrate_dl(lstm_test, lstm_fgsm , "LSTM defense FGSM testing FGSM attack", X_fgsm, Y_fgsm)
dnn_model, lstm_model = load_models('dnn_fgsm.h5','lstm_fgsm.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_fgsm)
log_results("Ensemble Deep Learning testing FGSM attack ", Y_fgsm, ensemble_dl)



detectionrate_ml(gnb_fgsm, "GaussianNB defense FGSM testing before zoo attack", X_zoo, Y_zoo)
detectionrate_ml(tdt_fgsm, "DT  defense FGSM testing before zoo Attack", X_zoo, Y_zoo)
detectionrate_ml(logic_fgsm, "LogisticRegression defense FGSM testing before zoo attack", X_zoo, Y_zoo)
detectionrate_ml(en_ml_fgsm, "Ensemble ML defense FGSM testing before zoo attack", X_zoo, Y_zoo)
detectionrate_dl(dnn_test, dnn_fgsm , "DNN defense FGSM testing before zoo attack", X_zoo, Y_zoo)
detectionrate_dl(lstm_test, lstm_fgsm , "LSTM defense FGSM testing before zoo attack", X_zoo, Y_zoo)
dnn_model, lstm_model = load_models('dnn_fgsm.h5','lstm_fgsm.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_zoo)
log_results("Ensemble Deep Learning defense FGSM testing zoo attack ", Y_zoo, ensemble_dl)






detectionrate_ml(gnb_fgsm, "GaussianNB defense FGSM testing before CW attack", X_cw, Y_cw)
detectionrate_ml(tdt_fgsm, "DT testing defense FGSM before CW Attack", X_cw, Y_cw)
detectionrate_ml(logic_fgsm, "LogisticRegression defense FGSM testing before CW attack", X_cw, Y_cw)
detectionrate_ml(en_ml_fgsm, "Ensemble ML defense FGSM testing before CW attack", X_cw, Y_cw)
detectionrate_dl(dnn_test, dnn_fgsm , "DNN defense FGSM testing before CW attack", X_cw, Y_cw)
detectionrate_dl(lstm_test, lstm_fgsm , "LSTM defense FGSM testing before CW attack", X_cw, Y_cw)
dnn_model, lstm_model = load_models('dnn_fgsm.h5','lstm_fgsm.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_cw)
log_results("Ensemble Deep Learning defense FGSM testing before CW  attack ", Y_cw, ensemble_dl)



detectionrate_ml(gnb_fgsm, "GaussianNB defense FGSM testing before deepfool attack", X_df, Y_df)
detectionrate_ml(tdt_fgsm, "DT testing defense FGSM before deepfool Attack", X_df, Y_df)
detectionrate_ml(logic_fgsm, "LogisticRegression defense FGSM testing before deepfool attack", X_df, Y_df)
detectionrate_ml(en_ml_fgsm, "Ensemble ML defense FGSM testing before deepfool attack", X_df, Y_df)
detectionrate_dl(dnn_test, dnn_fgsm , "DNN defense FGSM testing before deepfool attack", X_df, Y_df)
detectionrate_dl(lstm_test, lstm_fgsm , "LSTM defense FGSM testing before deepfool attack", X_df, Y_df)
dnn_model, lstm_model = load_models('dnn_fgsm.h5','lstm_fgsm.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_df)
log_results("Ensemble Deep Learning defense FGSM before deepfool Attack ", Y_df, ensemble_dl)




#---------------------------- CW ATTACK-------------------------
detectionrate_ml(gnb_cw, "GaussianNB defense CW testing before CW attack", X_cw, Y_cw)
detectionrate_ml(tdt_cw, "DT testing defense CW before CW Attack", X_cw, Y_cw)
detectionrate_ml(logic_cw, "LogisticRegression defense CW testing before CW attack", X_cw, Y_cw)
detectionrate_ml(en_ml_cw, "Ensemble ML defense CW testing before CW attack", X_cw, Y_cw)
detectionrate_dl(dnn_test, dnn_cw , "DNN defense CW testing before CW attack", X_cw, Y_cw)
detectionrate_dl(lstm_test, lstm_cw , "LSTM defense CW testing before CW attack", X_cw, Y_cw)
dnn_model, lstm_model = load_models('dnn_cw.h5','lstm_cw.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_cw)
log_results("Ensemble Deep Learning defense CW testing before CW  attack ", Y_cw, ensemble_dl)



detectionrate_ml(gnb_cw, "GaussianNB defense CW testing before FGSM attack", X_fgsm, Y_fgsm)
detectionrate_ml(tdt_cw, "DT testing defense CW before FGSM Attack", X_fgsm, Y_fgsm)
detectionrate_ml(logic_cw, "LogisticRegression defense CW testing before FGSM attack", X_fgsm, Y_fgsm)
detectionrate_ml(en_ml_cw, "Ensemble ML defense CW testing before FGSM attack", X_fgsm, Y_fgsm)
detectionrate_dl(dnn_test, dnn_cw , "DNN defense CW testing before FGSM attack", X_fgsm, Y_fgsm)
detectionrate_dl(lstm_test, lstm_cw , "LSTM defense CW testing before FGSM attack", X_fgsm, Y_fgsm)
dnn_model, lstm_model = load_models('dnn_cw.h5','lstm_cw.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_fgsm)
log_results("Ensemble Deep Learning defense CW testing before FGSM  attack ", Y_fgsm, ensemble_dl)



detectionrate_ml(gnb_cw, "GaussianNB defense CW testing before DeepFool attack", X_df, Y_df)
detectionrate_ml(tdt_cw, "DT testing defense CW before DeepFool Attack", X_df, Y_df)
detectionrate_ml(logic_cw, "LogisticRegression defense CW testing before DeepFool attack", X_df, Y_df)
detectionrate_ml(en_ml_cw, "Ensemble ML defense CW testing before DeepFool attack", X_df, Y_df)
detectionrate_dl(dnn_test, dnn_cw , "DNN defense CW testing before DeepFool attack", X_df, Y_df)
detectionrate_dl(lstm_test, lstm_cw , "LSTM defense CW testing before DeepFool attack", X_df, Y_df)
dnn_model, lstm_model = load_models('dnn_cw.h5','lstm_cw.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_df)
log_results("Ensemble Deep Learning defense CW testing before DeepFool  attack ", Y_df, ensemble_dl)

detectionrate_ml(gnb_cw, "GaussianNB defense CW testing before ZOO attack", X_zoo, Y_zoo)
detectionrate_ml(tdt_cw, "DT testing defense CW before ZOO Attack", X_zoo, Y_zoo)
detectionrate_ml(logic_cw, "LogisticRegression defense CW testing before ZOO attack", X_zoo, Y_zoo)
detectionrate_ml(en_ml_cw, "Ensemble ML defense CW testing before ZOO attack", X_zoo, Y_zoo)
detectionrate_dl(dnn_test, dnn_cw , "DNN defense CW testing before ZOO attack", X_zoo, Y_zoo)
detectionrate_dl(lstm_test, lstm_cw , "LSTM defense CW testing before ZOO attack", X_zoo, Y_zoo)
dnn_model, lstm_model = load_models('dnn_cw.h5','lstm_cw.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_zoo)
log_results("Ensemble Deep Learning defense CW testing before ZOO  attack ", Y_zoo, ensemble_dl)





##----------------------------- DEEPFOOL--------------------------------------------





detectionrate_ml(gnb_df, "GaussianNB defense DeepFool testing before DeepFool attack", X_df, Y_df)
detectionrate_ml(tdt_df, "DT testing defense DeepFool before DeepFool Attack", X_df, Y_df)
detectionrate_ml(logic_df, "LogisticRegression defense DeepFool testing before DeepFool attack", X_df, Y_df)
detectionrate_ml(en_ml_df, "Ensemble ML defense DeepFool testing before DeepFool attack", X_df, Y_df)
detectionrate_dl(dnn_test, dnn_df, "DNN defense DeepFool testing before DeepFool attack", X_df, Y_df)
detectionrate_dl(lstm_test, lstm_df, "LSTM defense DeepFool testing before DeepFool attack", X_df, Y_df)
dnn_model, lstm_model = load_models('dnn_df.h5','lstm_df.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_df)
log_results("Ensemble Deep Learning defense DeepFool testing before DeepFool attack ", Y_df, ensemble_dl)




detectionrate_ml(gnb_df, "GaussianNB defense DeepFool testing before FGSM attack", X_fgsm, Y_fgsm)
detectionrate_ml(tdt_df, "DT testing defense DeepFool before FGSM Attack", X_fgsm, Y_fgsm)
detectionrate_ml(logic_df, "LogisticRegression defense DeepFool testing before FGSM attack", X_fgsm, Y_fgsm)
detectionrate_ml(en_ml_df, "Ensemble ML defense DeepFool testing before FGSM attack", X_fgsm, Y_fgsm)
detectionrate_dl(dnn_test, dnn_df, "DNN defense DeepFool testing before FGSM attack", X_fgsm, Y_fgsm)
detectionrate_dl(lstm_test, lstm_df, "LSTM defense DeepFool testing before FGSM attack", X_fgsm, Y_fgsm)
dnn_model, lstm_model = load_models('dnn_df.h5','lstm_df.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_fgsm)
log_results("Ensemble Deep Learning defense DeepFool testing before FGSM attack ", Y_fgsm, ensemble_dl)


detectionrate_ml(gnb_df, "GaussianNB defense DeepFool testing before ZOO attack", X_zoo, Y_zoo)
detectionrate_ml(tdt_df, "DT testing defense DeepFool before ZOO Attack", X_zoo, Y_zoo)
detectionrate_ml(logic_df, "LogisticRegression defense DeepFool testing before ZOO attack", X_zoo, Y_zoo)
detectionrate_ml(en_ml_df, "Ensemble ML defense DeepFool testing before ZOO attack", X_zoo, Y_zoo)
detectionrate_dl(dnn_test, dnn_df, "DNN defense DeepFool testing before ZOO attack", X_zoo, Y_zoo)
detectionrate_dl(lstm_test, lstm_df, "LSTM defense DeepFool testing before ZOO attack", X_zoo, Y_zoo)
dnn_model, lstm_model = load_models('dnn_df.h5','lstm_df.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_zoo)
log_results("Ensemble Deep Learning defense DeepFool testing before ZOO attack ", Y_zoo, ensemble_dl)

detectionrate_ml(gnb_df, "GaussianNB defense DeepFool testing before CW attack", X_zoo, Y_zoo)
detectionrate_ml(tdt_df, "DT testing defense DeepFool before CW Attack", X_zoo, Y_zoo)
detectionrate_ml(logic_df, "LogisticRegression defense DeepFool testing before CW attack", X_zoo, Y_zoo)
detectionrate_ml(en_ml_df, "Ensemble ML defense DeepFool testing before CW attack", X_zoo, Y_zoo)
detectionrate_dl(dnn_test, dnn_df, "DNN defense DeepFool testing before CW attack", X_zoo, Y_zoo)
detectionrate_dl(lstm_test, lstm_df, "LSTM defense DeepFool testing before CW attack", X_zoo, Y_zoo)
dnn_model, lstm_model = load_models('dnn_df.h5','lstm_df.h5')
ensemble_dl = ensemble_voting_predict((dnn_model, lstm_model), X_zoo)
log_results("Ensemble Deep Learning defense DeepFool testing before CW attack ", Y_zoo, ensemble_dl)

