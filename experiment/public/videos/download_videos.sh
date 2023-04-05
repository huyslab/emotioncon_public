for i in $(seq -f "%04g" 1 2185)
do
	curl -O "https://s3-us-west-1.amazonaws.com/emogifs/mp4_noname/$i.mp4"
done
