# Mukhyansh

This repository contains the code,data and models of the paper titled [***"Mukhyansh: A Headline Generation Dataset for Indic Languages"***](https://arxiv.org/abs/2311.17743) published in the 37th Pacific Asia Conference on
Language, Information and Computation (PACLIC-2023)

## Table of Contents

- [Mukhyansh](#Mukhyansh)
  - [Dataset](#dataset)
  - [Installation](#installation)
  - [Models](#models)
  - [Benchmarks](#benchmarks)
  - [License](#license)
  - [Citation](#citation)

## Dataset

***Disclaimer: You must agree to the [license](#license) and terms of use before using the dataset.***

Please click [here](https://drive.google.com/drive/folders/1PYUgWMqELhVbQ_nJ7EtpYo_R1xm7XM6y?usp=sharing) to download the language-wise datasets.

We are releasing the Mukhyansh, an extensive multilingual dataset, tailored for Indian language headline generation. Comprising an impressive collection of over 3.39 million article-headline pairs, Mukhyansh spans across eight prominent Indian languages, namely Telugu, Tamil, Kannada, Malayalam, Hindi, Bengali, Marathi, and Gujarati.

All dataset files are in `.jsonl` format i.e. one JSON per line. Here is an example from `Telugu` dataset.

```
  {"id": 263,
 "url": "https://www.ap7am.com/flash-news-682791/modi-coronavirus-cases-in-india",
 "text": "ప్రధాన మంత్రి నరేంద్ర మోదీ మన్‌ కీ బాత్‌లో ప్రజలతో మాట్లాడుతున్నారు. కరోనాపై పోరులో భాగంగా లాక్‌డౌన్‌ వంటి అసాధారణ నిర్ణయాలు తీసుకోవాల్సి వస్తోందని, ప్రజలు ఇబ్బందులు ఎదుర్కొంటున్నారని, తనను క్షమించాలని వ్యాఖ్యానించారు. తనపై కొందరు ఆగ్రహంతో ఉన్నారని తనకు తెలుసని అన్నారు. అయినప్పటికీ, కరోనా వ్యాప్తిని అరికట్టడానికి ఈ కఠిన చర్యలు తీసుకోకతప్పదని చెప్పారు.\nముఖ్యంగా పేద ప్రజలు ఇబ్బందులు ఎదుర్కొంటున్నారని  మోదీ గుర్తు చేశారు. ప్రస్తుతం మనం ఎదుర్కొంటున్నది జీవన్మరణ సమస్య అయినందువల్లే కఠిన నిర్ణయం తీసుకున్నామని చెప్పారు. వైరస్ వ్యాప్తి ప్రారంభమైన రోజుల్లో చర్యలు తీసుకుంటేనే కరోనాను తొలగించవచ్చని అన్నారు. వైరస్ వ్యాప్తిని అరికట్టాలంటే దేశ ప్రజలు కొన్ని రోజులు లక్ష్మణ రేఖ దాటొద్దని వ్యాఖ్యానించారు.",
 "title": "ప్రజలు నన్ను క్షమించండి: 'మన్‌ కీ బాత్‌'లో ప్రధాని మోదీ",
 "category": "state"}

```

***Train-Dev-Test splits counts of each language:***

| Language   | ISO 639-1 Code | Train  | Dev   | Test  | Total |
|------------|----------------|--------|-------|-------|-------|
| Telugu     | te             | 825,372 | 82,571 | 9,179 | 917,122 |
| Tamil      | ta             | 298,543 | 26,539 | 6,626 | 331,708 |
| Kannada    | kn             | 304,122 | 27,044 | 6,757 | 337,923 |
| Malayalam  | ml             | 283,555 | 25,190 | 6,327 | 315,072 |
| Hindi      | hi             | 540,568 | 48,042 | 12,013 | 600,623 |
| Bengali    | bn             | 253,139 | 22,514 | 5,620 | 281,273 |
| Marathi    | mr             | 301,001 | 26,751 | 6,690 | 334,442 |
| Gujarati   | gu             | 248,367 | 22,073 | 5,518 | 275,958 |


***Category wise statistics of Mukhyansh:***

| News Category   | te     | ta    | kn    | ml    | hi     | bn    | mr    | gu    |
|-----------------|--------|-------|-------|-------|--------|-------|-------|-------|
| state           | 698,059 | 133,599 | 163,857 | 144,491 | -      | 143,804 | 184,045 | 123,183 |
| national        | 91,787  | 80,711  | 61,170  | 92,833  | 314,528 | 42,913  | 72,182  | 53,248  |
| entertainment   | 59,244  | 31,265  | 22,697  | 14,939  | 80,202  | 31,470  | 2,819   | 19,710  |
| international   | 24,262  | 29,463  | 26,092  | 34,008  | 29,668  | 20,552  | 15,347  | 37,682  |
| sports          | 19,933  | 29,186  | 18,775  | 10,204  | 78,190  | 30,676  | 29,947  | 19,337  |
| business        | 13,495  | 12,874  | 8,747   | 3,446   | 60,524  | 775    | 10,379  | 21,884  |
| crime           | 8,917   | 6,656  | 7,541   | 7,064   | 8,052   | -      | 16,489  | -      |
| covid           | 1,425   | 6,470  | 14,147  | 4,348   | -      | 4,205  | -      | -      |
| politics        | -      | 4,484  | 5,816   | 843    | 29,459  | 346    | 3,234   | -      |
| other           | -      | -      | 9,081   | 2,896   | -      | 6,532  | -      | 914    |


## Installation
To use this code, you need to have Python 3.7.11 installed. You can install the required Python packages using pip:

```bash
pip3 install -r requirements.txt
```

## Models
We used [huggingface transformers](https://github.com/huggingface/transformers) of version 4.25.0.
Please click [here]() to download the language-wise model-checkpoints.
### Fine-tune mT5-small Model
To fine-tune the mT5-small model, run the following command:

```bash
python3 run_summarization.py \
    --model_name_or_path google/mt5-small \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --source_prefix "summarize: " \
    --train_file <path to the '.jsonl' or '.csv' file> \
    --validation_file <path to the '.jsonl' or '.csv' file> \
    --test_file <path to the '.jsonl' or '.csv' file> \
    --max_target_length <Specify the maximum target sequence length> \
    --output_dir <path to the output directory> \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --num_train_epochs 10 \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --overwrite_output_dir True \
    --predict_with_generate $@ 2>&1>./hg_mt5_log.txt

```

### Fine-tune IndicBARTSS Model
To fine-tune the IndicBARTSS model, run the following command:

```bash
python3 run_summarization.py \
    --model_name_or_path ai4bharat/IndicBARTSS \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --source_prefix "summarize: " \
    --lang_id '<2te>' \
    --train_file <path to the '.jsonl' or '.csv' file> \
    --validation_file <path to the '.jsonl' or '.csv' file> \
    --test_file <path to the '.jsonl' or '.csv' file> \
    --max_target_length <Specify the maximum target sequence length> \
    --output_dir <path to the output directory> \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --num_train_epochs 10 \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --overwrite_output_dir True \
    --predict_with_generate $@ 2>&1>./hg_ssib_log.txt

```


***Note: For fine-tuning IndicBARTSS model, you should include `--lang_id` argument in the above command. For example, if you are doing it for Telugu language the `lang_id` will be `<2te>`. Here is the list of language id's: `<2te>`,`<2ta>`,`<2kn>`,`<2ml>`,`<2hi>`,`<2bn>`,`<2mr>`,`<2gu>`.***

## Benchmarks

ROUGE-L scores of various baseline models of Mukhyansh for each language. We use the [Multilingual ROUGE metric](https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring) from [xl-sum](https://github.com/csebuetnlp/xl-sum/tree/master).

| Language | FastText+LSTM | FastText+GRU | BPEmb+GRU | mT5-small | IndicBARTSS |
|---------|--------------|-------------|----------|----------|------|
| te      | 32.02        | 32.70       | 29.31    | 38.35    | 37.33|
| ta      | 32.20        | 31.26       | 32.04    | 41.18    | 41.16|
| kn      | 25.25        | 22.84       | 23.60    | 33.34    | 32.59|
| ml      | 28.17        | 23.44       | 25.36    | 34.63    | 32.04|
| hi      | 29.50        | 28.45       | 28.94    | 33.65    | 36.18|
| bn      | 17.47        | 14.90       | 9.84     | 21.56    | 22.04|
| mr      | 16.83        | 14.04       | 17.54    | 26.41    | 27.08|
| gu      | 14.84        | 9.48        | 14.94    | 20.43    | 23.05|
|         |              |             |          |          |      |
|***Avg***| 24.54        | 22.14       | 22.70    | 31.19    | 31.43|

***Hyper-parameters:***

| Parameters         | Seq-Seq(FastText) | Seq-Seq(BPEmb) | mT5-small | IndicBARTSS |
|--------------------|-------------------|----------------|-----------|------------|
| Max Source Length  | 200               | 300            | 1024      | 1024       |
| Max Target Length  | 20                | 30             | 30        | 30         |
| Vocabulary Size    | 40000             | 40000          | 250112    | 64000      |
| Beam Width         | 5                 | 5              | 4         | 4          |
| Batch Size         | 16                | 16             | 16        | 16         |
| Optimizer          | Adam              | Adam           | Adam      | Adam       |
| Learning Rate      | 1e-4              | 1e-4           | 5e-5      | 5e-5       |
| (GPU, CPU)         | (1,10)            | (1,10)         | (4,40)    | (4,40)     |



## License
Contents of this repository are restricted to only non-commercial research purposes under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). Copyright of the dataset contents belongs to the original copyright holders.

## Citation
If you use any of the datasets, models or code modules, please cite the following paper:

```
@article{madasu2023mukhyansh,
  title={Mukhyansh: A Headline Generation Dataset for Indic Languages},
  author={Madasu, Lokesh and Kanumolu, Gopichand and Surange, Nirmal and Shrivastava, Manish},
  journal={arXiv preprint arXiv:2311.17743},
  url= "https://arxiv.org/abs/2311.17743",
  year={2023}
}

```
