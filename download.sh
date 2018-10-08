apt-get install -y wget unzip
wget --no-check-certificate -N https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip
unzip -o Banana_Linux_NoVis.zip
mv ./Banana_Linux_NoVis/Banana.x86_64 .
rm -rf Banana_Linux_NoVis.zip ./Banana_Linux_NoVis