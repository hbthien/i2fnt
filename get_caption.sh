if [ $# -eq 0 ]
then 
  echo "path or url to image is missing"
  echo "example: ./get_caption.sh ./imgs/bikes.jpg"
  exit 3
fi

INPUTFILE=$1
echo $INPUTFILE | grep '^https\{0,1\}://'
if [ $? -eq 0 ]
then
  TMPFILE="./imgs/$(cat /proc/sys/kernel/random/uuid).jpg"
  wget -nv -O $TMPFILE "$INPUTFILE"
  INPUTFILE=$TMPFILE
fi

echo "Processing $INPUTFILE"

# Run inference to generate captions.¬
im2txt/bazel-bin/im2txt/run_inference --checkpoint_path="./pretrained_model/model2.ckpt-2000000" --vocab_file="./pretrained_model/word_counts.txt" --input_files="$INPUTFILE"

if [ -v TMPFILE ]
then
  echo "Removing tmp file $TMPFILE"
  rm "$TMPFILE"
fi
