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
echo $mo->toString($a);

$depth = 2;
$num_heads = 2;
$mha = new MultiHeadAttention($K,$nn,$depth,$num_heads);

$target = $g->Variable($K->ones([8,8,16]));
$source = $g->Variable($K->ones([8,4,16]));

[$output,$weights] = $mha($source,$source,$target,mask:null);

echo implode(',',$output->shape())."\n";
echo implode(',',$weights->shape())."\n";
