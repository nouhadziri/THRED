#!/bin/bash

BOLD='\033[1m'
RED='\033[0;31m'
BROWN='\033[0;33m'
NORMAL='\033[0m'


while [[ $# > 1 ]]
do
	key="$1"

	case $key in 
		-b|--batch-size)
		BATCH_SIZE="$2"
		shift # past argument
		;;
		-o|--out-dir)
		OUT_DIR="$2"
		shift # past argument
		;;
		--max-words)
		MAX_WORDS="$2"
		shift # past argument
		;;
		--max-chars)
		MAX_CHARS="$2"
		shift # past argument
		;;
		--min-words)
		MIN_WORDS="$2"
		shift # past argument
		;;
		--min-chars)
		MIN_CHARS="$2"
		shift # past argument
		;;
		-t|--subreddits)
		SUBREDDIT_FILE="$2"
		shift # past argument
		;;
		-c|--comments)
		COMMENTS_FILE="$2"
		shift # past argument
		;;
		-s|--submissions)
		SUBMISSIONS_FILE="$2"
		shift # past argument
		;;
		-k|--skip-lines)
		SKIP_LINES="$2"
		shift # past argument
		;;
		-l|--log)
		LOG_FILE="$2"
		shift # past argument
		;;
    		*)
	        # unknown option
    		;;
	esac
	shift # past argument or value
done

# Check requirements
if [ -z "$COMMENTS_FILE" ] && [ -z "$SUBMISSIONS_FILE" ]
then
	echo -e "${RED}An input compressed file must be provided either using -c (for comments) or -s (for submissions)${NORMAL}"
	exit 1
elif [ ! -z "$COMMENTS_FILE" ]; then
    filename="${COMMENTS_FILE##*/}"
    ext="${COMMENTS_FILE##*.}"
elif [ ! -z "$SUBMISSIONS_FILE" ]; then
    filename="${SUBMISSIONS_FILE##*/}"
    ext="${SUBMISSIONS_FILE##*.}"
fi

filename="${filename%.*}"
ext=${ext,,}
if [ "$ext" != "bz2" ] && [ "$ext" != "xz" ] && [ $ext != "bzip2" ]; then
    echo -e "${RED}The input file must be either bz2 or xz, but it is ${ext}${NORMAL}"
    exit 1
fi

PYTHON_CMD="python3"
cmd_check=$(command -v python3)
if [ ! -z "$cmd_check" ]; then
    version=$(python3 --version 2>&1 | cut -f2 -d ' ')
    version=${version%%.*}
    if [ "$version" != "3" ]; then
        PYTHON_CMD=""
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    command -v python >/dev/null 2>&1 || { echo -e "${RED}Python command (version 3) not found${NORMAL}"; exit 1; }
    version=$(python --version 2>&1 | cut -f2 -d ' ')
    version=${version%%.*}
    if [ "$version" != "3" ]; then
        echo -e "${RED}Make sure you have a Python version 3 command in the PATH${NORMAL}"
        exit 1
    fi
    PYTHON_CMD="python"
fi

if [ ! -f "thred/corpora/reddit/reddit_parser.py" ]
then
	if [ -f "../thred/corpora/reddit/reddit_parser.py" ]
	then
		cd ..
	else
		echo -e "${RED}Please go to the project base directory.${NORMAL}"
		exit 1
	fi
fi


# Optional args
BATCH_ARG=""
if [ ! -z "$BATCH_SIZE" ]; then
    BATCH_ARG="--batch_size $BATCH_SIZE"
fi

MAXW_ARG=""
if [ ! -z "$MAX_WORDS" ]; then
    MAXW_ARG="--max_words $MAX_WORDS"
fi

MINW_ARG=""
if [ ! -z "$MIN_WORDS" ]; then
    MINW_ARG="--min_words $MIN_WORDS"
fi

MAXC_ARG=""
if [ ! -z "$MAX_CHARS" ]; then
    MAXC_ARG="--max_chars $MAX_CHARS"
fi

MINC_ARG=""
if [ ! -z "$MIN_CHARS" ]; then
    MINC_ARG="--min_chars $MIN_CHARS"
fi

SKIP_ARG=""
if [ ! -z "$SKIP_LINES" ]; then
    SKIP_ARG="--skip_lines $SKIP_LINES"
fi

if [ -z "$SUBREDDIT_FILE" ]; then
	SUBREDDIT_FILE="thred/corpora/reddit/subreddit_whitelist.txt"
fi

if [ -f "$LOG_FILE" ]; then
	rm -f "$LOG_FILE"
	echo -e "${BROWN}Existing log file removed${NORMAL}"
fi

COMMENTS_ARG=""
SUBMISSIONS_ARG=""
XZ_FILE=""
if [ ! -z "$COMMENTS_FILE" ]
then
    if [ "$ext" == "xz" ]; then
        COMMENTS_ARG="--comments_stream"
        XZ_FILE="$COMMENTS_FILE"
    else
        COMMENTS_ARG="--comments_file $COMMENTS_FILE"
    fi

    # finding the containing directory is taken from https://stackoverflow.com/a/40700120
    if [ -z "$OUT_DIR" ]; then
        OUT_DIR="$(dirname -- "$(readlink -f -- "$COMMENTS_FILE")")"
    fi
else
    if [ "$ext" == "xz" ]; then
        SUBMISSIONS_ARG="--submissions_stream"
        XZ_FILE="$SUBMISSIONS_FILE"
    else
        SUBMISSIONS_ARG="--submissions_file $SUBMISSIONS_FILE"
    fi

    if [ -z "$OUT_DIR" ]; then
        OUT_DIR="$(dirname -- "$(readlink -f -- "$SUBMISSIONS_FILE")")"
    fi
fi

export PYTHONPATH="."

rnd=$(date | md5sum | head -c 4)
crash_file=".reddit_crash.$rnd"

if [ "$ext" == "xz" ]; then
    nohup sh -c "xz -d $XZ_FILE -c | $PYTHON_CMD -u thred/corpora/reddit/reddit_parser.py --out_dir $OUT_DIR --output_prefix $filename --subreddits $SUBREDDIT_FILE -r $crash_file $COMMENTS_ARG $SUBMISSIONS_ARG $BATCH_ARG $SKIP_ARG $MAXW_ARG $MAXC_ARG $MINW_ARG $MINC_ARG" >$LOG_FILE 2>&1 < /dev/null &
else
    nohup $PYTHON_CMD -u thred/corpora/reddit/reddit_parser.py --out_dir $OUT_DIR --output_prefix $filename --subreddits $SUBREDDIT_FILE -r $crash_file $COMMENTS_ARG $SUBMISSIONS_ARG $BATCH_ARG $SKIP_ARG $MAXW_ARG $MAXC_ARG $MINW_ARG $MINC_ARG >$LOG_FILE 2>&1 < /dev/null &
fi

echo -e "${BOLD}Crash file set to ${crash_file}${NORMAL}"
echo -e "${BOLD}Running... Check out log file ${LOG_FILE}${NORMAL}"
