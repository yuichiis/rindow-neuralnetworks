<?php

include __DIR__."/vendor/autoload.php";

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use function Rindow\Math\Matrix\R;

$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$plt = new Plot(null,$mo);

$K = $nn->backend();

//$x = $K->array([
//    [1,2,3,4],
//    [1,2,3,4],
//]);
//$y = $K->array([
//    [1,0,0,0],
//    [0,1,0,0],
//    [0,0,1,0],
//    [0,0,0,1],
//]);
//$c = $K->array([
//    [1,1,1,1],
//    [1,1,1,1],
//]);
//$z = $K->gemm($x,$y,c:$c);
//echo "z".$mo->shapeToString($z->shape()).":".$mo->toString($z,indent:true)."\n";

$mask = $K->array([
    1,
    0,
]);
$inputs = $K->array([
    [1,2,3,4],
    [1,2,3,4],
]);
$kernel = $K->array([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16],
]);
$bias = $K->array([
    1,1,1,1
]);
$mask = $K->array([
    1,
    0,
]);
$inputs = $K->mul($inputs,$mask,trans:true);
echo "masked inputs".$mo->shapeToString($inputs->shape()).":".$mo->toString($inputs,indent:true)."\n";

$outputs = $K->batch_gemm($inputs,$kernel,beta:1,c:$bias);
echo "z".$mo->shapeToString($outputs->shape()).":".$mo->toString($outputs,indent:true)."\n";
