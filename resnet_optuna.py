import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import Counter
from torchvision.utils import make_grid
import torch.nn.functional as F
import torchvision.models as models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
import optuna
from torch.utils.data import random_split
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import os
import csv

torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def load_dataset(dataset_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    return dataset


def draw_histogram_num_images(class_labels, counts):
    sorted_indices = np.argsort(class_labels)
    class_labels = np.array(class_labels)[sorted_indices]
    counts = np.array(counts)[sorted_indices]

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_labels, counts, align='center')
    plt.xlabel('Class Name')
    plt.ylabel('Number of Images')
    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 yval, count, ha='center', va='bottom')
    plt.title('Number of Images per Class')
    plt.savefig('train_images_class.png')
    plt.close()
    plt.clf()

def calculate_avg_metrics(all_labels, all_preds, average):
    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_precision = precision_score(
        all_labels, all_preds, average=average, zero_division=0)
    overall_recall = recall_score(
        all_labels, all_preds, average=average, zero_division=0)
    overall_f1 = f1_score(all_labels, all_preds,
                          average=average, zero_division=0)
    return overall_accuracy, overall_precision, overall_recall, overall_f1


def calculate_class_metrics(class_labels, class_preds):
    accuracy = accuracy_score(class_labels, class_preds)
    precision = precision_score(class_labels, class_preds, zero_division=0)
    recall = recall_score(class_labels, class_preds, zero_division=0)
    f1 = f1_score(class_labels, class_preds, zero_division=0)
    return accuracy, precision, recall, f1

def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    absolute_errors = np.abs(y_true - y_pred)
    mae = np.mean(absolute_errors)
    return mae

def draw_train_val_curve(train_losses, val_losses, val_accuracies, val_micro_aurocs,optuna_ops):
    # Plotting the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Training and Validation Losses per Epoch")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(optuna_ops+'/train_val_losses_p_epoch.png')
    plt.close()
    plt.clf()

    plt.figure(figsize=(10, 6))
    val_accuracies_cpu = [acc.cpu().numpy() for acc in val_accuracies]
    plt.plot(val_accuracies_cpu, label='Validation Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(optuna_ops+'/val_acc_p_epoch.png')
    plt.close()
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(val_micro_aurocs, label='Micro-average AUROC (Training)')
    plt.title('Micro-average AUROC per Epoch (Training)')
    plt.xlabel('Epochs')
    plt.ylabel('AUROC')
    plt.legend()
    plt.savefig(optuna_ops+'/val_micro_auroc_p_epoch.png')
    plt.close()
    plt.clf()

def test(weights,num_classes,test_loader,class_names,optuna_ops):
    model = resnet50(weights=weights)
    # 첫 번째 레이어 수정
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # 마지막 레이어 수정
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(optuna_ops+'/auroc.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dir_path = optuna_ops+'/test/'
    os.makedirs(dir_path, exist_ok=True)

    all_preds = []
    all_labels = []
    f1_scores = []
    all_proba = []

    class_labels = [[] for _ in range(num_classes)]
    class_preds = [[] for _ in range(num_classes)]
    class_probas = [[] for _ in range(num_classes)]
    class_probas_idx = [[] for _ in range(num_classes)]

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            proba_predictions = F.softmax(outputs, dim=1)
            all_proba.extend(proba_predictions.cpu().numpy())

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                proba = np.array(proba_predictions[i].cpu().numpy())
                class_labels[label].append(label)
                class_preds[label].append(pred)
                class_probas[label].append(proba)
                class_probas_idx[label].append(proba[label])
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_proba = np.array(all_proba)
                

    # calculate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,annot_kws={"size": 20})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(dir_path+'confusion_mat.png')
    plt.show()

    # Plot the ROC curve for each class
    plt.figure(figsize=(10, 10))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    class_labels_list = []
    class_probas_list = []

    table_rows = []
    total_class_acc = 0
    class_top2_recall_list = []
    top2_recall_total = 0
    balance_acc_total = 0
    total_mae = 0

    binary_class_labels = np.zeros((len(class_labels), num_classes))
    for i in range(len(class_labels)):
        for label in class_labels[i]:
            binary_class_labels[i, label] = 1

    top_k_acc_csv = dir_path+'top_k_pred.csv'
    total_macro_auroc = 0

    with open(top_k_acc_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Calculate metrics for each class

        for class_index in range(num_classes):  # Replace num_classes with the actual number of classes
            class_preds_bin = (all_preds == class_index)
            class_labels_bin = (all_labels == class_index)
            class_probas_bin = all_proba[:, class_index]
            for class_label in class_labels_bin:
                class_labels_list.append(class_label)
            for class_proba in class_probas_bin:
                class_probas_list.append(class_proba)

            second_max_list = []
            for array in class_probas[class_index]:
                second_max = np.partition(array,-2)[-2]
                second_max_idx = np.where(array == second_max)[0][0]
                second_max_list.append(second_max_idx)
            
            mae = mean_absolute_error(class_labels[class_index],class_preds[class_index])
            total_mae += mae

            csv_writer.writerow(["class_labels", class_labels[class_index]])
            csv_writer.writerow(["class_preds", class_preds[class_index]])
            csv_writer.writerow(["second_max_list", second_max_list])
            csv_writer.writerow(["mae",mae])
            csv_writer.writerow([])
            # Calculate the ROC curve
            fpr[class_index], tpr[class_index], _ = roc_curve(
                class_labels_bin, class_probas_bin)
            # Calculate the area under the ROC curve (AUC)
            roc_auc[class_index] = auc(fpr[class_index], tpr[class_index])
            total_macro_auroc += roc_auc[class_index]

            accuracy = cm[class_index][class_index] / np.sum(cm[class_index])
            balance_acc_total += accuracy
            
            accuracy, precision, recall, f1 = calculate_class_metrics(
            class_labels_bin, class_preds_bin)
            f1_scores.append(f1)
            class_name = class_names[class_index]
            total_class_acc+=accuracy
            count_top2_idx = np.sum(np.array(second_max_list) == class_index)
            top2_recall = (cm[class_index][class_index] + count_top2_idx) / np.sum(cm[class_index])
            top2_recall_total += top2_recall

            class_top2_recall_list.append(top2_recall)
            table_rows.append([class_name, roc_auc[class_index], mae, accuracy, precision, recall, f1, top2_recall])

    avg_mae = total_mae / num_classes
    avg_acc = total_class_acc / num_classes

    # Print the table
    fpr["micro"], tpr["micro"], _ = roc_curve(class_labels_list, class_probas_list)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc["macro"] = total_macro_auroc / num_classes

    all_fpr = np.unique(np.concatenate(
        [fpr[class_index] for class_index in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for class_index in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[class_index], tpr[class_index])

    mean_tpr = mean_tpr/num_classes

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.4f})'
                ''.format(roc_auc["micro"]),
            color='navy', linestyle=':', linewidth=4)

    for class_index in range(num_classes):
        plt.plot(fpr[class_index], tpr[class_index],
                label='ROC curve[{0}]:{1} (area = {2:0.4f})'
                ''.format(class_index, class_names[class_index], roc_auc[class_index]))


    micro_accuracy, micro_precision, micro_recall, micro_f1 = calculate_avg_metrics(all_labels, all_preds, "micro")
    _, macro_precision, macro_recall, macro_f1 = calculate_avg_metrics(all_labels, all_preds, "macro")

    # Calculate micro and macro metrics
    micro_metrics = [roc_auc['micro'],"", micro_accuracy, micro_precision, micro_recall, micro_f1]
    macro_metrics = [roc_auc["macro"],avg_mae, avg_acc, macro_precision, macro_recall, macro_f1]

    # Append micro and macro metrics to table_rows
    table_rows.append(["Macro",*macro_metrics])
    table_rows.append(["Micro", *micro_metrics])


    table_headers = ["Class", "Auroc", "MAE", "Accuracy", "Precision", "Recall", "F1 Score"]
    # print(tabulate(table_rows, headers=table_headers, tablefmt="pretty"))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for Each Class')
    plt.legend()
    plt.savefig(dir_path+'auroc.png')

    ##
    csv_file_path = dir_path+'metrics_results.csv'
    csv_class_names = class_names


    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(["Class", "Auroc", "MAE", "Accuracy", "Precision", "Recall", "F1 Score"])
        # Write class-wise metrics
        for class_index in range(num_classes):
            csv_writer.writerow([
                class_names[class_index],
                "{:.3f}".format(table_rows[class_index][1]),  # auroc
                "{:.3f}".format(table_rows[class_index][2]),  # mae
                "{:.3f}".format(table_rows[class_index][3]),  # precision
                "{:.3f}".format(table_rows[class_index][4]),  # recall
                "{:.3f}".format(table_rows[class_index][5]),  # f1
                "{:.3f}".format(table_rows[class_index][6]),  # top2 recall
            ])
            
        csv_writer.writerow([
            'macro',
            "{:.3f}".format(roc_auc["macro"]),  # auroc
            "{:.3f}".format(macro_metrics[1]),  # mae
            "{:.3f}".format(macro_metrics[2]),  # precision
            "{:.3f}".format(macro_metrics[3]),  # recall
            "{:.3f}".format(macro_metrics[4]),  # f1
            "{:.3f}".format(macro_metrics[5]),  # top2 recall
        ])
        csv_writer.writerow([
            'micro',
            "{:.3f}".format(micro_metrics[0]),  # auroc
            (micro_metrics[1]),  # mae
            "{:.3f}".format(micro_metrics[2]),  # precision
            "{:.3f}".format(micro_metrics[3]),  # recall
            "{:.3f}".format(micro_metrics[4]),  # f1
            "{:.3f}".format(micro_metrics[5]),  # top2 recall
        ])

    csv_file.close()

    return micro_metrics[0]

def objective(trial):
    # Define the search space for hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 4, 8)
    weight_decay = trial.suggest_float('weight_decay', 0.0001, 0.001, log=True)

    
    train_dir =  "/home/eu2goo/Emotional/train_test_split/train/"
    val_dir =  "/home/eu2goo/Emotional/train_test_split/val/"
    test_dir =  "/home/eu2goo/Emotional/train_test_split/test/"
    
    train_dataset = load_dataset(train_dir)
    val_dataset = load_dataset(val_dir)
    test_dataset = load_dataset(test_dir)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    class_names = train_dataset.classes
    num_classes = len(train_dataset.classes)
    
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    # 첫 번째 레이어 수정
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # 마지막 레이어 수정
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optuna_ops= "lr"+str(learning_rate)+"_bs"+str(batch_size)+"_wd"+str(weight_decay)
    os.makedirs(optuna_ops, exist_ok=True)
    with open('output.txt', 'a') as log_file:
        log_file.write(f"{optuna_ops}\n")
        

    log_file_path = optuna_ops+"/training_log.txt"
    best_val_loss = float('inf')
    best_val_acc = float(0.0)
    best_val_auroc = float(0.0)
    num_epochs= 100
    p_acc_counter =0 
    p_loss_counter = 0
    p_auroc_counter = 0

    train_losses =[]
    val_losses=[]
    val_accuracies =[]
    all_preds=[]
    all_labels =[]
    val_micro_aurocs = []
    log_file_path = optuna_ops+'/training_log.txt'
    csv_file_path = optuna_ops+'/training_metrics.csv'

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        val_corrects =0
        val_labels = []
        val_probas = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device),labels.to(device)
                outputs = model(images)
                loss = criterion(outputs,labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs,1)
                val_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())
                val_labels.extend(labels.view(-1).cpu().numpy())
                val_probas.extend(F.softmax(outputs, dim=1).cpu().numpy())
            val_probas = np.array(val_probas)
            val_labels = np.array(val_labels)
            
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len (val_loader.dataset)
            val_accuracy = val_corrects.double() / len(val_loader.dataset)
            val_auroc = roc_auc_score(val_labels,val_probas,average='micro',multi_class='ovr')
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_micro_aurocs.append(val_auroc)
            draw_train_val_curve(train_losses,val_losses,val_accuracies,val_micro_aurocs,optuna_ops)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), optuna_ops+'/valloss.pth')
                p_loss_counter = 0
            else:
                p_loss_counter+=1
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(model.state_dict(), optuna_ops+'/valacc.pth')
                p_acc_counter = 0
            else:
                p_acc_counter+=1
            if val_auroc >  best_val_auroc:
                best_val_auroc = val_auroc
                torch.save(model.state_dict(), optuna_ops+'/auroc.pth')
                p_auroc_counter = 0
            else:
                p_auroc_counter+=1
            torch.save(model.state_dict(), optuna_ops+'/epoch.pth')

            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val auroc:{val_auroc}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy}, patience counter(acc,loss,auroc): {p_acc_counter},{p_loss_counter},{p_auroc_counter}\n")
        
            with open(csv_file_path, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([epoch + 1, train_loss, val_auroc, val_loss, val_accuracy])
            trial.report(best_val_auroc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    test_auroc = test(weights,num_classes,test_loader,class_names,optuna_ops)
    return test_auroc

# Create a study object and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)
best_param_path = '/home/eu2goo/Emotional/code/0527/best_params.txt'
with open(best_param_path, 'a') as log_file:
    log_file.write(f"{best_params}")
