curdir="/Users/bi/gitprojects/IDepthDataset/"
# Replace /Users/bi with /home/anhvth8 
new_dir="/home/anhvth8/gitprojects/IDepthDataset/"

echo sync from $curdir to dms:$new_dir
# CHeck if first args is pull
if [ "$1" = "pull" ];
  then
    # Pull from dms
    cmd="rsync -avz -e ssh --exclude-from 'exclude.txt' anhvth8@dms:$new_dir $curdir"
    echo $cmd
    eval $cmd
elif [ "$1" = "push" ];
  then
    # rsync over ssh
    cmd="rsync -avz -e ssh --exclude-from 'exclude.txt' $curdir anhvth8@dms:$new_dir"
    echo $cmd
    eval $cmd
elif [ "$1" = "run" ];
  then
    ./dms.sh push
    python_cmd="/home/anhvth8/.conda/envs/swin/bin/python ${@:2}"
    echo $python_cmd

    cmd="ssh dms -t \"cd ${new_dir} && ${python_cmd}\""
    echo $cmd
    eval $cmd
else
    echo "Usage: dms.sh [pull|push|ssh]"
fi
