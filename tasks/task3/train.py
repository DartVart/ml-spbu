from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

PREFIX = 'get a title: '
DATASET_NAME = 'IlyaGusev/gazeta'
MODEL_NAME = 'cointegrated/rut5-small'


def preprocess_dataset(
    items,
    tokenizer,
    prefix,
):
    inputs = [prefix + text for text in items["summary"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=items["title"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def get_test_train_dataset(
    dataset_name,
    tokenizer,
    prefix,
    train_size=30000,
    test_size=5000,
    seed=666,
):
    dataset = load_dataset(dataset_name)
    shuffled_dataset = dataset['train'].shuffle(seed=seed)

    train_dataset = shuffled_dataset.select(range(train_size))
    test_dataset = shuffled_dataset.select(range(train_size, train_size + test_size))

    _preprocess_dataset = lambda items: preprocess_dataset(items, tokenizer, prefix)

    tokenized_train = train_dataset.map(_preprocess_dataset, batched=True)
    tokenized_test = test_dataset.map(_preprocess_dataset, batched=True)

    return tokenized_train, tokenized_test


def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer


if __name__ == '__main__':
    print('Training started...')

    model, tokenizer = get_model(MODEL_NAME)

    print('Pretrained model is loaded!')

    train_dataset, test_dataset = get_test_train_dataset(DATASET_NAME, tokenizer, PREFIX)

    print('Train and tests datasets are prepared!')

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./result_models",  # where the model predictions and checkpoints will be written
        eval_steps=500,
        logging_steps=500,
        learning_rate=2e-5,  # the initial learning rate for AdamW optimizer
        per_device_train_batch_size=16,  # the batch size per GPU/CPU for training
        per_device_eval_batch_size=16,  # the batch size per GPU/CPU for evaluation
        weight_decay=0.01,  # the weight decay to all layers except all bias and LayerNorm weights in AdamW optimizer
        save_total_limit=3,  # it will limit the total amount of checkpoints
        num_train_epochs=3,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        predict_with_generate=True,  # whether to use generate to calculate generative metrics (ROUGE, BLEU)
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print('Start training...')

    trainer.train()

    print('The training is over! The model has been saved.')
