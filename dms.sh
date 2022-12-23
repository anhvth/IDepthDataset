local_dir="/Users/bi/gitprojects/IDepthDataset/"
remote_dir="/home/anhvth8/gitprojects/IDepthDataset/"
REMOTE_MACHINE=anhvth8@dms
echo sync from $local_dir to dms:$remote_dir

if [ "$1" = "pull" ];
  then
    # Pull from dms
    cmd="rsync -avz -e ssh --exclude-from 'exclude.txt' $REMOTE_MACHINE:$remote_dir $local_dir ${@:2}}"
    echo $cmd
    eval $cmd
elif [ "$1" = "push" ];
  then
    # rsync over ssh
    cmd="rsync -avz -e ssh --exclude-from 'exclude.txt' $local_dir $REMOTE_MACHINE:$remote_dir ${@:2}"
    echo $cmd
    eval $cmd
else
    # If the first arg is python change it to "/home/anhvth8/.conda/envs/swin/bin/python"
    app=$2
    if [ "$app" = "python" ];
      then
        app="/home/anhvth8/.conda/envs/swin/bin/python"
    fi
    # args from 3rd arg
    args="${@:3}"
    ./dms.sh push
    clear
    dms_cmd="$app $args"
    echo $dms_cmd

    cmd="ssh dms -t \"cd ${remote_dir} && ${dms_cmd}\""
    eval $cmd
    ./dms.sh pull
fi
