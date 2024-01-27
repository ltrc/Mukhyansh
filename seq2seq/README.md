## Train and Evaluate FastText+GRU model

Please check the following example(for `kannada`) snippet to train the FastText+GRU model.

```bash
python3 seq2seq.py \
    --vocab_csv 'data/kannada_vocab.csv' \
    --train_file 'data/kannada_train.jsonl' \
    --dev_file 'data/kannada_dev.jsonl' \
    --test_file 'data/kannada_test.jsonl' \
    --model_weights './model_weights.h5' \
    --test_outputs_file './test_prediction.txt' \
    --fastText_embeddings_file './cc.kn.300.bin' \
    --use_bpe False \
    --language 'kannada' \
    --do_train True \
    --do_test True \
    --train_batch_size 32 \
    --val_batch_size 16 \
    --test_batch_size 1 \
    --epochs 1 \
    --rnn_type 'gru' \
    --max_input_length 200 \
    --max_target_length 20 \
    --beam_size 5 \
    --beamsearch_length_penalty 0.1

```
## Train and Evaluate BPEmb+GRU model

Here is the example(for `kannada`) snippet to train the BPE+GRU model.

```bash
python3 seq2seq.py \
    --vocab_csv 'data/kannada_bpe_vocab.csv' \
    --train_file 'data/kannada_train.jsonl' \
    --dev_file 'data/kannada_dev.jsonl' \
    --test_file 'data/kannada_test.jsonl' \
    --model_weights './model_weights.h5' \
    --test_outputs_file './test_prediction.txt' \
    --use_bpe True \
    --bpe_lang_code 'kn' \
    --bpe_vocab_size 50000 \
    --language 'kannada' \
    --do_train True \
    --do_test True \
    --train_batch_size 32 \
    --val_batch_size 16 \
    --test_batch_size 1 \
    --epochs 1 \
    --rnn_type 'gru' \
    --max_input_length 300 \
    --max_target_length 30 \
    --beam_size 5 \
    --beamsearch_length_penalty 0.1

```

## Arguments

- `vocab_csv` (file_path): This is a .csv file that should contain two columns: 1) Index and 2) tokens. It contains the top 40,000 frequent tokens/words in the corpus.

- `train_file` (file_path): This is a '.jsonl' file that contains training data in the following format: {'id': 1, 'title': "Title text...", text: "News article text..."}.

- `dev_file` (file_path): This is the path to a '.jsonl' file containing validation data.

- `test_file` (file_path, optional): This is the path to a '.jsonl' file containing test data.

- `model_weights` (checkpoint path): This is the path to the 'model_weight.h5' file. If you have already saved a model checkpoint, provide that path; otherwise, specify the output path to save the model_weights.h5 file (e.g., './model_weights.h5').

- `test_outputs_file` (file_path, optional): This is the path to an output text file (e.g., outputs.txt) that will contain the predicted headlines of test news articles.

- `fastText_embeddings_file` (file_path, optional): This is the path to the fastText embedding model file. For instance, for Telugu, the path would be './cc.te.300.bin'.

- `use_bpe` (bool): Set this to 'True' or 'False' to determine whether to use BPE embeddings or FastText embeddings.

- `bpe_lang_code` (str, optional): Specify the language code to load the corresponding BPE embeddings.

- `bpe_vocab_size` (int, optional, defaults to 50000): Specify the BPE vocabulary size, for example, 1000, 5000, 10000, 50000, 100000, etc.

- `language` (str, optional): Specify the language name for multilingual rouge score calculation. For example, ['telugu', 'tamil', 'kannada', 'malayalam', 'hindi', 'bengali', 'marathi', 'gujarati'].

- `do_train` (bool): Set this to 'True' or 'False' to indicate whether to train the model.

- `do_test` (bool): Set this to 'True' or 'False' to indicate whether to test the model.

- `train_batch_size` (int, optional, defaults to 8): Specify the training batch size, for example, [4, 8, 16, etc].

- `val_batch_size` (int, optional, defaults to 8): Specify the validation batch size, for example, [4, 8, 16, etc].

- `test_batch_size` (int, optional, defaults to 1): Specify the test batch size.

- `epochs` (int): Specify the number of epochs for training the model.

- `rnn_type` (str): Specify which RNN type to use, such as `gru` or `lstm`.

- `max_input_length` (int): Specify the maximum total input sequence length. Sequences longer than this will be truncated, and sequences shorter will be padded.

- `max_target_length` (int): Specify the maximum total sequence length for the target text.

- `beam_size` (int, optional, defaults to 5): Specify the number of beams to use for evaluation. This argument will be used during evaluation/prediction.

- `beamsearch_length_penalty` (float, optional, defaults to 0.0): Specify the length penalty (alpha value) for beam search.
