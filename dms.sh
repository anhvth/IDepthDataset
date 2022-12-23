curdir="/Users/bi/gitprojects/IDepthDataset/"
new_dir="/home/anhvth8/gitprojects/IDepthDataset/"
echo sync from $curdir to dms:$new_dir

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
else
    ./dms.sh push
    clear
    dms_cmd="${@:2}"
    echo $dms_cmd

    cmd="ssh dms -t \"cd ${new_dir} && ${dms_cmd}\""
    eval $cmd
fi
