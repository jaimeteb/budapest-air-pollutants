#!/bin/bash

echo 'Downloading site files'
wget -nv -m -k -E -l 3 -t 5 -w 3 --random-wait https://jaimeteb.my.canva.site/

echo 'Updating site files'
rm -rf site/
mv jaimeteb.my.canva.site/ site/

echo 'Fix HTML'
tidy -i -o site/index_tmp.html site/index.html
perl -0777 -pe 's/<script nonce.*?<\/script>//gs' site/index_tmp.html > site/index.html
rm site/index_tmp.html

echo 'Done'
exit 0
