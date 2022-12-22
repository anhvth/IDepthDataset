curdir="/Users/bi/gitprojects/coherent-lidar/"
# Replace /Users/bi with /home/anhvth8 
new_dir=${curdir//\/Users\/bi/\/home\/anhvth8}
echo sync from $curdir to dms:$new_dir
# rsync over ssh
cmd="rsync -avz -e ssh --exclude-from 'exclude.txt' $curdir anhvth8@dms:$new_dir --delete"
echo $cmd
eval $cmd

# ssh to dms
echo cd $new_dir
ssh dms
