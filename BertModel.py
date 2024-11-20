import pandas as pd
import torch
import time
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def tokenize_data(data, tokenizer, max_len=128):
    #Uses given tokenizer to tokenize the data
    def tokenize_function(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_len)

    dataset = Dataset.from_pandas(data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    #Returns tokenized data
    return tokenized_dataset

def train_model(train_file, test_file):
    #Load the data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    #Load BERT tokenizer and BERT model
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    #Preprocess the data
    train_dataset = tokenize_data(train_data, tokenizer)
    test_dataset = tokenize_data(test_data, tokenizer)

    #Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
    )

    #Define a trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = test_dataset,
        tokenizer=tokenizer,
    )

    #Train the model
    trainer.train()
    #Save the model
    trainer.save("review_bert_model")
    return model

def evaluate_confusion_matrix(cm):
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return TP, TN, FP, FN, accuracy

def evaluate_model(model, eval_dataset, batch_size=32, device=None):
    start_time = time.time()
    #Determine the device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Move model to the selected device
    model = model.to(device)
    model.eval()

    #Creates a DataLoader for the evaluation dataset
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
    eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    #Initialize arrays for label and predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        count = 1
        for batch in eval_dataloader:
            print(f"\rprogress: {count}/{len(eval_dataloader)}", end=' ')
            #Move batch data to the device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            #Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            #Get predictions
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            count += 1

    inference_time = time.time() - start_time
    #Calculate evaluation metrics
    cm = confusion_matrix(all_labels, all_preds)
    TP, TN, FP, FN, accuracy = evaluate_confusion_matrix(cm)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

    print("True Positives: ", TP)
    print("True Negatives: ", TN)
    print("False Positives: ", FP)
    print("False Negatives: ", FN)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nf1: {f1}")
    print("Inference Time: ", inference_time)

# Replace with the paths to your CSV files
train_file = "train_amazon.csv"
test_file = "test_amazon.csv"

try:
    #Checks if model is found
    model = DistilBertForSequenceClassification.from_pretrained("review_bert_model")
    print("Model Found")
except:
    #If model cannot be found new model is trained
    print("Model Not Found, Training Pre-Trained Model")
    model = train_model(train_file, test_file)

print("Tokenizing Test Data")
test_dataset = tokenize_data(pd.read_csv(test_file), DistilBertTokenizer.from_pretrained("distilbert-base-uncased"))
print("Evaluating Model")
evaluate_model(model, test_dataset)