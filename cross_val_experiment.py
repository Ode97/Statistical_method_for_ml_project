#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import os
import csv

def load_dataset(dataset_dir):
        data = []
        labels = []
        class_dirs = os.listdir(dataset_dir)
        class_dict = {cls: i for i, cls in enumerate(class_dirs)}

        for cls in class_dirs:
            class_dir = os.path.join(dataset_dir, cls)
            class_label = class_dict[cls]

            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)

                image = Image.open(image_path).convert("L")
                image = image.resize((32, 32)) 
                image = np.array(image) 
                data.append(image)
                labels.append(class_label)

        data = np.array(data)
        data = data/255
        labels = np.array(labels)

        return data, labels


tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

def model1(lr):
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])
    
    return model

def model2(lr):
    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])
    
    return model


def model3(lr):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])
    
    return model

def model4(lr):
    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy',metrics=['accuracy'])
    
    return model


def plot_risks_matrix(model, risks):
    # Organizza i dati in una matrice
    risks_matrix = np.array(risks).reshape(len(learning_rates), len(batch_size))

    fig, ax = plt.subplots()

    # Valori sugli assi
    ax.set_xticks(np.arange(len(learning_rates)))
    ax.set_yticks(np.arange(len(batch_size)))
    ax.set_xticklabels(learning_rates)
    ax.set_yticklabels(batch_size)

    # Etichette sugli assi
    plt.xlabel("Learning Rate")
    plt.ylabel("Batch Size")
    plt.title(f"{model.__name__} - Risks for Different Learning Rates and Batch Sizes")

    for i in range(len(learning_rates)):
        for j in range(len(batch_size)):
            text = ax.text(j, i, f"{risks_matrix[i, j]:.4f}", ha="center", va="center", color="w")

    im = ax.imshow(risks_matrix, cmap="coolwarm", aspect='auto')
    
    vmin, vmax = risks_matrix.min(), risks_matrix.max()
    im.set_clim(vmin, vmax)

    plt.colorbar(im)
    
    plt.savefig(f'graph/{model.__name__}_risks.png')
    plt.show()

def plot_training_validation_loss(model, mean_train_losses, mean_val_losses):
    
    x_labels = [f'LR={lr}, BS={bs}' for bs in batch_size for lr in learning_rates]
    x_values = np.arange(len(x_labels))

    # Create the bar plot
    plt.figure(figsize=(13, 3))  # Optional: Adjust the figure size
    bar_width = 0.2

    for i in range(len(learning_rates) * len(batch_size)):
        label = x_labels[i]
        plt.bar(x_values[i] - bar_width/2, mean_train_losses[i],
                width=bar_width, label=f'Training ({label})', color='blue')
        plt.bar(x_values[i] + bar_width/2, mean_val_losses[i],
                width=bar_width, label=f'Validation ({label})', color='orange')
        
        plt.text(x_values[i] - bar_width/2, max(mean_train_losses[i], mean_val_losses[i]) + 0.02, f'{mean_train_losses[i]:.2f}', ha='center', color='blue')
        plt.text(x_values[i] + bar_width/2, max(mean_train_losses[i], mean_val_losses[i]) + 0.02, f'{mean_val_losses[i]:.2f}', ha='center', color='orange')


    plt.xticks(x_values, x_labels, rotation = 30, ha='center')  # Label with LR and BS values with rotation
    plt.ylabel('Mean Score')
    plt.title(f'{model.__name__} - Mean Training and Validation Loss for Different Hyperparameters')
    plt.ylim(0, 1)  # Adjust the y-axis limits as needed
    #plt.legend()
    plt.tight_layout()  # Optional: Adjusts subplot parameters for a better layout
    plt.savefig(f'graph/{model.__name__}_loss.png')
    plt.show()

def cross_val(model, X, Y, batch_size, learning_rate, patience):
    
    print(f"{model.__name__} batch_size: {batch_size} learning_rate: {learning_rate}")
    
    scores = []
    zero_one_losses = []
    train_loss = []
    val_loss = []
    
    epochs = []
    train_data = []
    val_data = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    splits = skf.split(X, Y)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        
    with tf.device('/gpu:0'):
        for fold, (train_index, val_index) in enumerate(splits):

            train_data = X[train_index]
            train_labels = Y[train_index]
            val_data = X[val_index]
            val_labels = Y[val_index]

            print(f"Fold {fold+1}/5")
            actual_model = model(learning_rate)
            history = actual_model.fit(train_data, train_labels, validation_data=(val_data, val_labels), callbacks=[early_stopping], batch_size = batch_size, epochs = 50, verbose=1)
            #score = actual_model.evaluate(val_data, val_labels)
            best_epoch = np.argmin(history.history["val_loss"])
            
            train_loss.append(history.history["loss"][best_epoch])
            val_loss.append(history.history["val_loss"][best_epoch])
            zero_one_loss = 1 - history.history["val_accuracy"][best_epoch]
    
            zero_one_losses.append(zero_one_loss)
        
            epochs.append(best_epoch)

    train_loss_mean = np.mean(train_loss)
    val_loss_mean = np.mean(val_loss)
    risk = np.mean(zero_one_losses)
    epochs_mean = np.mean(epochs)
    print(f"risk estimate: {risk}")
    
    return train_loss_mean, val_loss_mean, risk, epochs_mean


def hyperparameter_test_cross_validation(model): 
    risks = []
    mean_train_losses = []
    mean_val_losses = []
    mean_epochs = []
    
    for b in batch_size:
        for lr in learning_rates:
            mean_train_loss, mean_val_loss, risk, epochs_mean = cross_val(model, X, Y, b, lr, patience)
            f = open("result/result.txt", "a")
            f.write(f"{model.__name__} epochs mean:{epochs_mean} learning rate:{lr} batch size: {b} risk: {risk}\n")
            f.close()
            risks.append(risk)
            mean_train_losses.append(mean_train_loss)
            mean_val_losses.append(mean_val_loss)
            mean_epochs.append(epochs_mean) 
            
    
    with open(f'result/{model.__name__}_risks.pkl', 'wb') as file:
        pickle.dump(risks , file)
    
    with open(f'result/{model.__name__}_val.pkl', 'wb') as file:
        pickle.dump(mean_val_losses , file)

    with open(f'result/{model.__name__}_train.pkl', 'wb') as file:
        pickle.dump(mean_train_losses , file)

    return mean_train_losses, mean_val_losses, risks, mean_epochs


def write_csv(models, risks, epochs):
    if os.path.exists("result/result.csv"):
        os.remove("result/result.csv")
    # Creare un file CSV e scrivere i risultati
    with open("result/result.csv", mode="w", newline="") as csv_file:
        fieldnames = ["Model", "Learning Rate", "Batch Size", "Epochs Mean", "Risk"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()

        for i in range(len(batch_size)):
            for j in range(len(learning_rates)):
                for m in range(len(models)):
                    model_name = models[m].__name__
                    epochs_mean = epochs[m][i * len(batch_size) + j]  # Calcola il valore corrispondente alle epoche medie
                    lr = learning_rates[j]
                    bs = batch_size[i]
                    risk = risks[m][i * len(batch_size) + j]

                writer.writerow({"Model": model_name, "Learning Rate": lr, "Batch Size": bs, "Epochs Mean": epochs_mean, "Risk": risk})


learning_rates = [0.01, 0.001, 0.0001]
batch_size = [32, 128, 256]
patience = 5


X, Y = load_dataset("data/dataset")
if os.path.exists("result/result.txt"):
    os.remove("result/result.txt")


model1_train, model1_val, model1_risks, model1_epochs = hyperparameter_test_cross_validation(model1)


plot_risks_matrix(model1, model1_risks)
plot_training_validation_loss(model1, model1_train, model1_val)


model2_trian, model2_val, model2_risks, model2_epochs = hyperparameter_test_cross_validation(model2)

plot_risks_matrix(model2, model2_risks)
plot_training_validation_loss(model2, model2_train, model2_val)


model3_train, model3_val, model3_risks, model3_epochs = hyperparameter_test_cross_validation(model3)

plot_risks_matrix(model3, model3_risks)
plot_training_validation_loss(model3, model3_train, model3_val)


model4_train, model4_val, model4_risks, model4_epochs = hyperparameter_test_cross_validation(model4)


plot_risks_matrix(model4, model4_risks)
plot_training_validation_loss(model4, model4_train, model4_val)


models = [model1, model2, model3, model4]
risks = [model1_risks, model2_risks, model3_risks, model4_risks]
epochs = [model1_epochs, model2_epochs, model3_epochs, model4_epochs]

write_csv(models, risks, epochs)






