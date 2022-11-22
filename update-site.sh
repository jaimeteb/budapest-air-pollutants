#!/bin/bash

wget -nv -m -k -E -l 3 -t 5 -w 3 --random-wait https://jaimeteb.my.canva.site/
rm -rf site/
mv jaimeteb.my.canva.site/ site/

exit 0
