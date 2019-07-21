[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5a6cfaad36294d27a8479b227627f1c7)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=nouhadziri/Dialogue-Generation-ANA&amp;utm_campaign=Badge_Grade)

This repository hosts the implementation of the paper "[Augmenting Neural Response Generation with Context-Aware Topical
Attention](https://arxiv.org/abs/1811.01063)".

# Topical Hierarchical Recurrent Encoder Decoder (THRED)
THRED is a multi-turn response generation system intended to produce contextual and topic-aware responses.
The codebase is evolved from the Tensorflow [NMT](https://github.com/tensorflow/nmt) repository.

__TL;DR__ Steps to create a dialogue agent using this framework:
 1. Download the Reddit Conversation Corpus from [here](https://bit.ly/2LwKVWX) (2.5GB download / 7.8GB after uncompressing, which contains triples extracted from Reddit). Please report errors/inappropriate content in the data [here](https://forms.gle/1WfWw5ABHx9GAaVV6).
 2. Install the dependencies using `conda env create -f thred_env.yml` (To use `pip`, see [Dependencies](#dependencies))
 3. Train the model using the following command (pretrained models will be published soon). Note that `MODEL_DIR` is a directory that the model will be saved into. We recommend to train on at least 2 GPUs, otherwise you can reduce the data size (by omitting conversations from the training file) and the model size (by modifying the config file).
 ```
 python -m thred --mode train --config conf/thred_medium.yml --model_dir <MODEL_DIR> \
--train_data <TRAIN_DATA> --dev_data <DEV_DATA> --test_data <TEST_DATA>
 ```
 4. Chat with the trained model using:
 ```
 python -m thred --mode interactive --model_dir <MODEL_DIR>
 ```

## Dependencies
 - Python >= 3.5 (Recommended: 3.6)
 - Tensorflow == 1.12.0
 - Tensorflow-Hub
 - SpaCy >= 2.1.0
 - pymagnitude
 - tqdm
 - redis<sup>1</sup>
 - mistune<sup>1</sup>
 - emot<sup>1</sup>
 - Gensim<sup>1</sup>
 - prompt-toolkit<sup>2</sup>

<sup>1</sup><sub><sup>*packages required only for parsing and cleaning the Reddit data.*</sup></sub>
<sup>2</sup><sub><sup>*used only for testing dialogue models in command-line interactive mode*</sup></sub>
 
To install the dependencies using `pip`, run `pip install -r requirements`.
And for Anaconda, run `conda env create -f thred_env.yml` (recommended).
Once done with the dependencies, run `pip install -e .` to install the thred package. 

## Data
Our Reddit dataset, which we call Reddit Conversation Corpus (RCC), is collected from 95 selected subreddits (listed [here](thred/corpora/reddit/subreddit_whitelist.txt)).
We processed Reddit for a 20 month-period ranging from November 2016 until August 2018 (excluding June 2017 and July 2017; we utilized these two months along with the October 2016 data to train an LDA model). Please see [here](thred/corpora/reddit) for the details of how the Reddit dataset is built including pre-processing and cleaning the raw Reddit files. The following table summarizes the RCC information:

| Corpus    	        | #train| #dev  | #test | Download | Download with topic words|
|----------	            |:-----:|:-----:|:-----:|:-----------|:-----------|
| 3 turns per line   	| 9.2M  | 508K  | 406K  | [download](https://bit.ly/2SqZThP) (773MB) | [download](https://bit.ly/2LwKVWX) (2.5GB) | 
| 4 turns per line	    | 4M    | 223K  | 178K  | [download](https://bit.ly/2Gm5gKm)  (442MB) | [download](https://bit.ly/30E6HeW) (1.2GB)
| 5 turns per line	    | 1.8M  | 100K  | 80K   | [download](https://bit.ly/2JStT29) (242MB) | [download](https://bit.ly/2JFmYKO) (594MB)

In the data files, each line corresponds to a single conversation where utterances are TAB-separated. The topic words appear after the last utterance separated also by a TAB.

Note that the 3-turns/4-turns/5-turns files contain similar content albeit with different number of utterances per line. They are all extracted from the same source. If you found any error or any inappropriate utterance in the data, please report your concerns [here](https://forms.gle/1WfWw5ABHx9GAaVV6).

### Embeddings
In the model config files (i.e., the YAML files in [conf](conf)), the embedding types can be either of the following: `glove840B`, `fastText`, `word2vec`, and `hub_word2vec`. For handling the pre-trained embedding vectors, we leverage [Pymagnitude](https://github.com/plasticityai/magnitude/) and [Tensorflow-Hub](https://tfhub.dev/).
Note that you can also use `random300` (300 refers to the dimension of embedding vectors and can be replaced by any arbitrary value) to learn vectors during training of the response generation models. The settings related to embedding models are provided in [word_embeddings.yml](conf/word_embeddings.yml). 


## Train
The training configuration should be defined in a YAML file similar to Tensorflow NMT.
Sample configurations for THRED and other baselines are provided [here](conf).

The implemented models are [Seq2Seq](https://arxiv.org/abs/1409.3215), [HRED](https://arxiv.org/abs/1605.06069), [Topic Aware-Seq2Seq](https://arxiv.org/abs/1606.08340), and THRED.

Note that while most of the parameters are common among the different models, some models may have additional parameters 
(e.g., topical models have `topic_words_per_utterance` and `boost_topic_gen_prob` parameters).

To train a model, run the following command:
```bash
python main.py --mode train --config <YAML_FILE> \
--train_data <TRAIN_DATA> --dev_data <DEV_DATA> --test_data <TEST_DATA> \
--model_dir <MODEL_DIR>
```
In `<MODEL_DIR>`, vocabulary files and Tensorflow model files are stored. Training can be resumed by executing:
```bash
python main.py --mode train --model_dir <MODEL_DIR>
```

## Test
With the following command, the model can be tested against the test dataset. 

```bash
python main.py --mode test --model_dir <MODEL_DIR> --test_data <TEST_DATA>
``` 
It is possible to override test parameters during testing.
These parameters are: beam width `--beam_width`, 
length penalty weight `--length_penalty_weight`, and sampling temperature `--sampling_temperature`.

A simple command line interface is implemented that allows you to converse with the learned model (Similar to test mode, the test parameters can be overrided too):
```bash
python main.py --mode interactive --model_dir <MODEL_DIR>
```
In the interactive mode, a pre-trained LDA model is required to feed the inferred topic words into the model. We trained an LDA model using Gensim on a Reddit corpus, collected for this purpose.
It can be downloaded from [here](https://s3.ca-central-1.amazonaws.com/ehsk-research/thred/pretrained_LDA_reddit.tgz).
The downloaded file should be uncompressed and passed to the program via `--lda_model_dir <DIR>`.  

## Citation
Please cite the following paper if you used our work in your research:
```text
@article{dziri2018augmenting,
  title={Augmenting Neural Response Generation with Context-Aware Topical Attention},
  author={Dziri, Nouha and Kamalloo, Ehsan and Mathewson, Kory W and Zaiane, Osmar R},
  journal={arXiv preprint arXiv:1811.01063},
  year={2018}
}
```
