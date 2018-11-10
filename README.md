This repository hosts the implementation of the paper "[Augmenting Neural Response Generation with Context-Aware Topical
Attention](https://arxiv.org/abs/1811.01063)".

# Topical Hierarchical Recurrent Encoder Decoder (THRED)
THRED is a multi-turn response generation system intended to produce contextual and topic-aware responses.
The codebase is evolved from the Tensorflow [NMT](https://github.com/tensorflow/nmt) repository.

## Dependencies
- Python >= 3.5
- Tensorflow >= 1.4.0
- Tensorflow-Hub
- SpaCy
- Gensim
- PyYAML
- tqdm
- redis<sup>1</sup>
- mistune<sup>1</sup>
- emot<sup>1</sup>
- prompt-toolkit<sup>2</sup>

<sup>1</sup><sub><sup>*packages required only for parsing and cleaning the Reddit data.*</sup></sub>
<sup>2</sup><sub><sup>*used only for testing dialogue models in command-line interactive mode*</sup></sub>
 
To install the dependencies using `pip`, run `pip install -r requirements`.
And for Anaconda, run `conda env create -f thred_env.yml` (recommended). 

## Data
Our Reddit dataset is collected from 95 selected subreddits (listed [here](corpora/reddit/subreddit_whitelist.txt)).
We processed Reddit for a 12 month-period ranging from December 2016 until December 2017 (excluding June and July; we utilized these two months to train an LDA model). Please see [here](corpora/reddit) for the details of how the Reddit dataset is built including pre-processing and cleaning the raw Reddit files.

In the data files, each line corresponds to a single conversation where utterances are tab-separated. Topic words appear after the last utterance by a delimiter '  |  ' (a vertical bar preceding and trailing two whitespaces).

#### Embeddings
First, pre-trained word embedding models should be downloaded by running the following Python script:
```bash
export PYTHONPATH="."; python util/get_embed.py
```
The script downloads and extracts the [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings file.
The output is stored in the direcctory `workspace/embeddings`.
Additionally, the following options are available:
<pre>
    -e, --embedding_type        glove or word2vec (default: glove)
    -d, --dimensions            #dimensions of embedding vectors (default: 300)
    -f, --embedding_file        In case of using a non-default embedding, 
                                you can provide an embedding file loadable by Gensim (default: None)
</pre>       

In the model config files (explained below), the default embedding types can be either of the following: `glove`, `word2vec`, and `tf_word2vec`.
Note that `tf_word2vec` refers to the pre-trained word2vec provided in Tensorflow Hub [Wiki words](https://tfhub.dev/google/Wiki-words-500-with-normalization/1).
**If you intend to use the embeddings from Tensorflow Hub, there is no need to run the above command.**  


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
```
@article{dziri2018augmenting,
  title={Augmenting Neural Response Generation with Context-Aware Topical Attention},
  author={Dziri, Nouha and Kamalloo, Ehsan and Mathewson, Kory W and Zaiane, Osmar R},
  journal={arXiv preprint arXiv:1811.01063},
  year={2018}
}
```
