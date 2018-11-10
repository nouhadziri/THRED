## Building Reddit corpus
Initially, Reddit monthly data needs to be downloaded from [here](https://files.pushshift.io/reddit/comments/) for comments and [here](https://files.pushshift.io/reddit/submissions/) for submissions.
The data is stored as either bzip2 or xz compressed files and there is no need to decompress them. 

Note that it may take several days (on commodity hardware) to parse a full month Reddit data. Also, extracting conversations may take several hours.


### Pre-processing
The first step is parsing and pre-processing the Reddit json file.
The pre-processing includes case folding, emojis and emoticons stripping, replacing urls with `<URL>` placeholder, and omitting non-English posts, and etc.
 How to run the parser is explained as follows: 

#### On Unix systems
You can run the following shell script (works only on Unix systems) in order to process the Reddit comments:
```bash
bin/reddit.sh -c <REDDIT_COMMENT_COMPRESSED_FILE> --log <LOG_FILE>
```  
For submissions files, replace `-c` argument with `-s <REDDIT_SUBMISSION_COMPRESSED_FILE>`.

Here is the complete list of arguments that can be passed to the script:
<pre>
    --max-chars        maximum number of characters in a message after normalization (default: 150)
    --max-words        maximum number of words in a message after normalization and disabled by default
    --min-chars        minimum number of characters in a message after normalization (default: 5)
    --min-words        minimum number of words in a message after normalization (default: 2)
    -b, --batch-size   the parsed data is flushed to disk when exceeding batch size (default: 100000)
    -k, --skip-lines   number of lines to skip from the input Reddit file (default: 0)
    -o, --out-dir      the output directory to store the generated files
                       (default: same directory as the input file)
    -t, --subreddits   list of accepted subreddits (default: corpora/reddit/subreddit_whitelist.txt)
</pre>

We put together a list of 95 subreddits in [here](corpora/reddit/subreddit_whitelist.txt).
The subreddit file is a tab-separated file including the Reddit URL (e.g., [/r/news](https://reddit.com/r/news) ) and subreddit id.

The script will generate two files with the same name as the input file: a text file containing the processed text and a csv file (whose name is trailed by "\_db") containing meta information associated with the Reddit post.

#### On Non-Unix systems

To be able to run the Reddit parser on other machines, you need to run [reddit_parser.py](corpora/reddit/reddit_parser.py) 
as the following if the input file is a bzip2 file (note that for submissions `--submissions_file` should be used instead of `--comments_file`):
```bash
export PYTHONPATH="."; python corpora/reddit/reddit_parser.py \
    --comments_file <COMMENT_BZ2_FILE> -t <SUBREDDITS_FILE> 
    --output_prefix <PREFIX_FOR_OUTPUT_FILE>
    [-o <OUTPUT_DIR> --max_chars <NUM> --max_words <NUM> --min_chars <NUM> --min_words <NUM>]
```

And as below, in case the input is a xz file:
```bash
export PYTHONPATH="."; xz -d -c <COMMENT_XZ_FILE> | python corpora/reddit/reddit_parser.py \
    --comments_stream -t <SUBREDDITS_FILE>
    --output_prefix <PREFIX_FOR_OUTPUT_FILE>
    [-o <OUTPUT_DIR> --max_chars <NUM> --max_words <NUM> --min_chars <NUM> --min_words <NUM>]
```
Similary, `--submissions_stream` for submissions xz files.

### Creating Dialogues
After pre-processing, we rebuild the conversation threads based on parent links (which is basically the reference post to which the current post replies) stored in the metadata csv file.
Each conversation thread is treated as a tree where the nodes represent posts and comments and an edge from parent to child denotes that the child is a reply to the parent.
In these trees, we consider paths from root to leaves as conversation samples. Run the following command to generate the dialogue samples: 
```bash
export PYTHONPATH="."; python corpora/reddit/reddit_dialogue.py build \
    --text_file <TEXT_FILE> --csv_file <CSV_FILE>
    --output <OUTPUT_FILE>
```

There is also an option to break down the conversations to a fixed number of utterances (e.g., our experiments has been run on converation triples):
```bash
export PYTHONPATH="."; python corpora/reddit/reddit_dialogue.py prepare \
    --dialogue_data <DIALOGUE_DATA> --num_turns <NUMBER_OF_TURNS>
```


### Adding Topic Words
Finally, we exploited a pre-trained LDA model to acquire topic words.
This is step is required only when the topical models (i.e., TA-Seq2Seq and THRED) are being used for training and testing.
Note that our pre-trained LDA models can be downloaded from [here](https://s3.ca-central-1.amazonaws.com/ehsk-research/thred/pretrained_LDA_reddit.tgz) (trained on Reddit) and [here](https://s3.ca-central-1.amazonaws.com/ehsk-research/thred/pretrained_LDA_opensubtitles.tgz) (trained on OpenSubtitles).

```bash
export PYTHONPATH="."; python topic_model/lda.py --mode infer --model_dir <LDA_MODEL_DIR> --dialogue_as_doc \
--test_data <TEST_DATA> --output <OUT_FILE>
```
 