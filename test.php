<?php
include __DIR__.'/samples/neural-machine-translation-with-transformer.php';

use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$K = $nn->backend();
$g = $nn->gradient();

$a = $mo->array(1);
echo $mo->toString($a)."\n";

//$batchSize = 5;
//$wordVectSize = 16;
//$depth = 4;
//$num_heads = 2;
//$inputLen = 3;
//$targetLen = 7;
//$mha = new MultiHeadAttention($K,$nn,$depth,$num_heads);
//
//$target = $g->Variable($K->ones([$batchSize,$targetLen,$wordVectSize]));
//$source = $g->Variable($K->ones([$batchSize,$inputLen,$wordVectSize]));
//$mask = $mo->ones([$batchSize,$inputLen],NDArray::int8);
//
//[$output,$weights] = $mha($source,$source,$target,mask:$mask);

//echo implode(',',$output->shape())."\n";
//echo implode(',',$weights->shape())."\n";

//$shape = [2,2,2];
//$a = $mo->arange((int)array_product($shape),NDArray::int32)->reshape($shape);
//$la = $mo->la();
//echo implode(',',$a->shape())."\n";
////echo $mo->toString($a,null,true)."\n";
//$a = $la->repeat($a,3,axis:-1);
//echo $mo->toString($a,null,true)."\n";
//echo implode(',',$a->shape())."\n";
//$enclayer = new EncoderLayer($K,$nn,
//
//);

//$x = $mo->array(
//    [[[1,2],
//      [0,0]],
//     [[0,0],
//      [0,0]]]
//);
//echo '==== input ===='."\n";
//echo $mo->toString($x,'%6.6f',true).'('.implode(',',$x->shape()).')'."\n";
//echo '==== layer norm rindow ===='."\n";
//$shape = $x->shape();
//$size = array_pop($shape);
//$mean = $K->mean($x,axis:-1);
//$mean = $K->expandDims($mean,axis:-1);
//echo $mo->toString($mean,'%6.6f',true).'('.implode(',',$mean->shape()).')'."\n";
//echo '===='."\n";
//$mean = $K->repeat($mean,$size,axis:-1);
//$mean = $K->squeeze($mean,axis:-1);
//echo $mo->toString($mean,'%6.6f',true).'('.implode(',',$mean->shape()).')'."\n";
//echo '===='."\n";
//$var = $K->mean($K->square($K->sub($x,$mean)),axis:-1);
//$var = $K->squeeze($K->repeat($K->expandDims($var,axis:-1),$size,axis:-1),axis:-1);
//echo '===='."\n";
//$norm = $K->div($K->sub($x,$mean),$K->sqrt($K->increment($var,1e-7)));
//echo $mo->toString($norm,'%6.6f',true).'('.implode(',',$norm->shape()).')'."\n";

//$bnorm = $nn->layers->BatchNormalization();
//$x = $mo->array(
//    [[[1,0],
//      [2,0]],
//     [[3,0],
//      [4,0]]]
//);
//echo '==== input ===='."\n";
//echo $mo->toString($x,'%6.6f',true).'('.implode(',',$x->shape()).')'."\n";
//echo '===='."\n";
//$norm = $bnorm($x,training:true);
//echo $mo->toString($norm,'%6.6f',true).'('.implode(',',$norm->shape()).')'."\n";
//########################################################
//$bnorm = $nn->layers->LayerNormalization();
//$x = $mo->array(
//    [[[1,2],
//      [3,4]],
//     [[0,0],
//      [0,0]]]
//);
//echo '==== input ===='."\n";
//echo $mo->toString($x,'%6.6f',true).'('.implode(',',$x->shape()).')'."\n";
//echo '===='."\n";
//$norm = $bnorm($x,training:true);
//echo $mo->toString($norm,'%6.6f',true).'('.implode(',',$norm->shape()).')'."\n";


//$enclayer = new EncoderLayer($K,$nn,
//  $wordVectSize=4, 
//  $num_heads=2,
//  $dff=8, // units
//  $dropout_rate=0.1);
//
//$batchSize = 2;
//$inputLen  = 8;
//$x = $mo->ones([$batchSize,$inputLen,$wordVectSize]);
//$mask = $mo->ones([$batchSize,$inputLen],NDArray::int8);
//$y = $enclayer($x,$training=true,$mask);

$enc = new Encoder(
  $K,
  $nn,
  $numLayers=2,
  $num_heads=2,
  $dff=4,
  $vocabSize=4,
  $wordVectSize=4,
  $units=4,
  $inputLength=8,
  $dropout_rate=0.1,
);
