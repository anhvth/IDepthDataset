local_dir="/Users/bi/gitprojects/IDepthDataset/"
remote_dir="/home/anhvth8/gitprojects/IDepthDataset/"
REMOTE_MACHINE=anhvth8@dms # The server make sure you can ssh without password to this machine, and rsync is installed on both machines
# echo sync from $local_dir to dms:$remote_dir
echo "Local dir" $local_dir
echo "Remote dir" $remote_dir

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
    # check if file changes in local using fswatch, if true then eval cmd
    fswatch -o $local_dir | xargs -n1 -I{} sh -c "$cmd"
else
    cur_dir=$(pwd)/
    # Replace local_dir by remote_dir, the dirs might contain special characters
    rmd=${cur_dir/$local_dir/$remote_dir}
    cmd='ssh -t dms  "cd ${rmd}; zsh "'
    clear
    eval $cmd
    # ./dms.sh pull
fi
