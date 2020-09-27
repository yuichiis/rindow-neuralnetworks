<?php
$url = 'http://www.manythings.org/anki/fra-eng.zip';
echo "downloading...\n";
$a = file_get_contents($url);
echo "writing...\n";
file_put_contents('fra-eng.zip',$a);
