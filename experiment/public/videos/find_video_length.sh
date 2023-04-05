files=$(find -E . -type f -regex "^.*.mp4")
declare -a length_array=()
declare -a size_array=()
declare -a video_list=()
for item in ${files[*]}
do
	x=$(ffprobe -i $item -show_entries format=duration -v quiet -of csv="p=0")
	y=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 $item)
	if test -z "$y"
	then
      		length_array+=-1'\n'
		size_array+='-1x-1\n'
	else
      		length_array+=$x'\n'
		size_array+=$y'\n'
	fi
	video_list+=$item
done
echo $length_array $size_array $video_list > length_size_videos.txt
